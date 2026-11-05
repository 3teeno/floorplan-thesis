import csv
import json
import os
import re
import time
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from urllib.parse import urljoin, urlparse, unquote
from urllib.robotparser import RobotFileParser

import requests
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Keywords that indicate an image is a floor plan rather than a photo.
# Extend this per language/site as needed.
FLOORPLAN_KEYWORDS = [
    "pohjapiirros", "pohjakuva", "pohjapiirustus",  # Finnish
    "floorplan", "floor-plan", "floor_plan", "plan view",  # English
    "planritning", "planlosning",  # Swedish
    "grundriss",  # German
]


class BaseScraper(ABC):
    """Shared scraping mechanics. Subclass per site."""

    def __init__(self, base_url: str, output_dir: str = "output",
                 request_delay: float = 2.0, max_retries: int = 3):
        self.base_url = base_url
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.images_dir = os.path.join(output_dir, "floorplan_images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.session = requests.Session()
        # A realistic user-agent avoids being blocked outright by some sites.
        # This does not bypass any access controls — sites that block
        # scraping via robots.txt or ToS should not be scraped regardless.
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        })

        # robots.txt for the target site, read once. enforce_robots (default
        # True) makes fetch() refuse any path disallowed for our user-agent.
        self.enforce_robots = True
        self._robots = RobotFileParser()
        self._robots.set_url(urljoin(self.base_url, "/robots.txt"))
        try:
            self._robots.read()
        except Exception as e:  # network hiccup reading robots -> fail open, but say so
            log.warning(f"Could not read {self.base_url}/robots.txt ({e}); proceeding without robots enforcement")
            self._robots = None

    def robots_allows(self, url: str) -> bool:
        if self._robots is None:
            return True
        return self._robots.can_fetch(self.session.headers["User-Agent"], url)

    def listing_key(self, url: str) -> str:
        """Stable id for a listing URL, used to skip already-scraped listings on
        re-runs. Oikotie/Etuovi both end their listing URLs with a numeric id."""
        return urlparse(url).path.rstrip("/").split("/")[-1]

    def fetch(self, url: str) -> requests.Response | None:
        """GET a URL with retries and rate limiting. Returns None on failure."""
        if self.enforce_robots and not self.robots_allows(url):
            log.error(f"robots.txt disallows {url} for this user-agent -- skipping")
            return None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=15)
                time.sleep(self.request_delay)
                if resp.status_code == 200:
                    return resp
                log.warning(f"Status {resp.status_code} for {url} (attempt {attempt})")
            except requests.RequestException as e:
                log.warning(f"Request failed for {url}: {e} (attempt {attempt})")
            time.sleep(self.request_delay * attempt)  # backoff
        log.error(f"Giving up on {url} after {self.max_retries} attempts")
        return None

    def is_floorplan_image(self, img_url: str, alt_text: str = "") -> bool:
        """Heuristic: does this image look like a floor plan, not a photo?"""
        haystack = f"{img_url} {alt_text}".lower()
        return any(kw in haystack for kw in FLOORPLAN_KEYWORDS)

    def download_image(self, img_url: str, filename: str) -> str | None:
        """Download and save an image, converting to RGB JPEG. Returns saved path or None.
        Skips the download if the file already exists (so re-runs are cheap)."""
        path = os.path.join(self.images_dir, filename)
        if os.path.exists(path):
            return path
        resp = self.fetch(img_url)
        if resp is None:
            return None
        try:
            image = Image.open(BytesIO(resp.content))
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            image.save(path, format="JPEG", quality=95)
            return path
        except Exception as e:
            log.warning(f"Could not process image {img_url}: {e}")
            return None

    @abstractmethod
    def get_listing_urls(self) -> list[str]:
        """Return every individual listing URL to visit. Site-specific."""
        raise NotImplementedError

    @abstractmethod
    def parse_listing(self, url: str, soup: BeautifulSoup) -> dict:
        """Extract structured fields + floor plan image URLs from one listing page. Site-specific."""
        raise NotImplementedError

    def _load_done(self) -> tuple[list[dict], set[str]]:
        """Read any listings.jsonl left by a previous run so we can resume."""
        jsonl_path = os.path.join(self.output_dir, "listings.jsonl")
        records, done = [], set()
        if os.path.exists(jsonl_path):
            with open(jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    records.append(rec)
                    done.add(str(rec.get("listing_id", "")))
            log.info(f"Resuming: {len(records)} listings already in {jsonl_path}")
        return records, done

    def _write_tables(self, records: list[dict]) -> pd.DataFrame:
        """(Re)write listings.csv + listings.xlsx from the full record list."""
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, "listings.csv"), index=False,
                  quoting=csv.QUOTE_MINIMAL)
        try:
            df.to_excel(os.path.join(self.output_dir, "listings.xlsx"), index=False)
        except Exception as e:  # openpyxl missing etc. -- csv is the source of truth
            log.warning(f"Could not write xlsx ({e}); listings.csv is up to date")
        return df

    def run(self, limit: int | None = None, only_with_floorplan: bool = False) -> pd.DataFrame:
        """Scrape listings. Resumes from output/listings.jsonl if present.

        limit: cap on the number of *new* listings fetched this run.
        only_with_floorplan: skip listings whose page had no floor plan image.
        """
        listing_urls = self.get_listing_urls()
        log.info(f"Found {len(listing_urls)} candidate listings")

        records, done = self._load_done()
        jsonl_path = os.path.join(self.output_dir, "listings.jsonl")
        jsonl = open(jsonl_path, "a", encoding="utf-8")

        fetched = 0
        for i, url in enumerate(listing_urls, 1):
            key = self.listing_key(url)
            if key in done:
                continue
            if limit is not None and fetched >= limit:
                log.info(f"Reached limit of {limit} new listings")
                break
            log.info(f"[{i}/{len(listing_urls)}] {url}")
            resp = self.fetch(url)
            if resp is None:
                continue
            fetched += 1
            soup = BeautifulSoup(resp.content, "html.parser")
            try:
                record = self.parse_listing(url, soup)
            except Exception as e:
                log.warning(f"Failed to parse {url}: {e}")
                continue
            record.setdefault("listing_id", key)
            record["source_url"] = url
            done.add(key)
            if only_with_floorplan and not record.get("floorplan_count"):
                log.info("  no floor plan on this listing -- skipping (only_with_floorplan)")
                continue
            records.append(record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()
            if len(records) % 20 == 0:
                self._write_tables(records)

        jsonl.close()
        df = self._write_tables(records)
        log.info(f"Saved {len(df)} listings -> {self.output_dir}/listings.csv (+ .xlsx, .jsonl)")
        return df


# Municipality names as they appear in oikotie.fi listing URL slugs once
# percent-decoded and lower-cased. Pass a subset (or your own list) as
# OikotieScraper(cities=[...]); the CLI --region switch picks one of the groups.
PIRKANMAA_CITIES = [
    "tampere", "nokia", "ylöjärvi", "kangasala", "lempäälä", "pirkkala",
    "vesilahti", "orivesi", "valkeakoski", "akaa", "urjala", "kylmäkoski",
    "hämeenkyrö", "ikaalinen", "parkano", "kihniö", "sastamala", "punkalaidun",
    "virrat", "ruovesi", "juupajoki", "mänttä-vilppula", "pälkäne",
]

# All 26 Uusimaa municipalities (Helsinki metropolitan region + surrounds).
UUSIMAA_CITIES = [
    "helsinki", "espoo", "vantaa", "kauniainen", "kirkkonummi", "kerava",
    "järvenpää", "tuusula", "nurmijärvi", "hyvinkää", "mäntsälä", "pornainen",
    "sipoo", "vihti", "lohja", "karkkila", "siuntio", "inkoo", "raasepori",
    "hanko", "porvoo", "askola", "myrskylä", "pukkila", "lapinjärvi", "loviisa",
]

REGIONS = {
    "pirkanmaa": PIRKANMAA_CITIES,
    "uusimaa": UUSIMAA_CITIES,
    "both": PIRKANMAA_CITIES + UUSIMAA_CITIES,
}


class OikotieScraper(BaseScraper):
    """Scraper for asunnot.oikotie.fi (for-sale apartments only).

    Chosen over etuovi.com for AI-run scraping: asunnot.oikotie.fi/robots.txt
    disallows only /api*, /card* (the partner-key-gated endpoints) and a
    handful of account/print paths -- none of which this class touches -- and
    has no AI-crawler-specific block like etuovi.com's. It publishes a sitemap
    for discovering listing pages, which get_listing_urls reads instead of
    crawling search-result pages.

    Every listing page embeds a `var otAsunnot={...}` JS object carrying the
    data the page's own frontend hydrates from: address, coordinates, build
    year (otAsunnot["analytics"]["apartmentBuildYear"]), room count, and an
    images list where the floor plan (when Oikotie found one) is tagged with a
    string containing "floorplan" -- seen as both "floorplan" and
    "ai_floorplan", so matched by substring. A schema.org `Apartment` ld+json
    block gives a second, clean source for rooms / size / postcode.

    Output columns are aimed at a floor-plan-vectorisation thesis dataset:
    location (address, postcode, district, lat/lon) + build year + room count
    + the floor plan image, which is exactly the metadata CubiCasa5K lacks.

    GDPR: listing pages carry the selling agent's name. It is dropped unless
    include_agent_names=True. The agency (company) name is kept -- that is a
    business, not personal data.
    """

    SITEMAP_INDEX = "https://asunnot.oikotie.fi/sitemaps/index.xml"
    LISTING_PATH_PREFIX = "/myytavat-asunnot/"  # for-sale housing only, not rentals/business premises
    STATE_MARKER = "var otAsunnot="

    def __init__(self, cities: list[str] | None = None, sitemap_shards: int | None = None,
                 include_agent_names: bool = False, **kwargs):
        """cities: keep only listings in these municipalities (see
        PIRKANMAA_CITIES); None keeps every for-sale listing found.
        sitemap_shards: how many sm_ad_*.xml shards to read (15 exist, each
        ~3k for-sale listings nationwide plus rentals). Defaults to all 15 when
        a city filter is set (the XML is cheap), else 1.
        include_agent_names: keep the selling agent's personal name (GDPR).
        """
        super().__init__(base_url="https://asunnot.oikotie.fi", **kwargs)
        self.cities = {c.casefold() for c in cities} if cities else None
        self.sitemap_shards = sitemap_shards if sitemap_shards is not None else (15 if self.cities else 1)
        self.include_agent_names = include_agent_names

    @staticmethod
    def _city_slug(url: str) -> str:
        """Municipality slug from a listing URL, percent-decoded and lower-cased.
        /myytavat-asunnot/<city>/<id> -> '<city>'; '' when the URL omits it."""
        parts = urlparse(unquote(url)).path.strip("/").split("/")
        return parts[1].casefold() if len(parts) >= 3 else ""

    def get_listing_urls(self) -> list[str]:
        resp = self.fetch(self.SITEMAP_INDEX)
        if resp is None:
            return []
        index_soup = BeautifulSoup(resp.content, "xml")
        ad_sitemaps = [loc.text for loc in index_soup.find_all("loc") if "sm_ad_" in loc.text]

        seen, urls, skipped_city = set(), [], 0
        for sitemap_url in ad_sitemaps[:self.sitemap_shards]:
            resp = self.fetch(sitemap_url)
            if resp is None:
                continue
            soup = BeautifulSoup(resp.content, "xml")
            for loc in soup.find_all("loc"):
                url = loc.text
                if urlparse(url).netloc != "asunnot.oikotie.fi":
                    continue
                if self.LISTING_PATH_PREFIX not in url or url in seen:
                    continue
                if self.cities is not None and self._city_slug(url) not in self.cities:
                    skipped_city += 1
                    continue
                seen.add(url)
                urls.append(url)
        shards = min(self.sitemap_shards, len(ad_sitemaps))
        msg = f"{len(urls)} for-sale listings across {shards} sitemap shard(s)"
        if self.cities is not None:
            msg += f" ({skipped_city} skipped by city filter: {sorted(self.cities)})"
        log.info(msg)
        return urls

    def _parse_state(self, soup: BeautifulSoup) -> dict:
        """The embedded `var otAsunnot={...}` object -- parsed with raw_decode
        rather than a regex-bounded match, since the object is followed by more
        JS statements on the same line, not a clean `};`.
        """
        for script in soup.find_all("script"):
            text = script.string or ""
            if self.STATE_MARKER not in text:
                continue
            start = text.find(self.STATE_MARKER) + len(self.STATE_MARKER)
            try:
                data, _ = json.JSONDecoder().raw_decode(text[start:])
                return data
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _ld_apartment(soup: BeautifulSoup) -> dict:
        """The schema.org Apartment/Product ld+json block, if present."""
        for sc in soup.find_all("script", type="application/ld+json"):
            try:
                d = json.loads(sc.string or "")
            except json.JSONDecodeError:
                continue
            t = d.get("@type")
            if t == "Apartment" or (isinstance(t, list) and "Apartment" in t):
                return d
        return {}

    def _details_grid(self, soup: BeautifulSoup) -> dict[str, str]:
        """Every label/value pair from the page's spec table (Rakennusvuosi,
        huoneiden lkm, kerros, ...) -- generic over whatever fields a given
        listing happens to show.
        """
        details = {}
        for dt in soup.select("dt.details-grid__item-title"):
            dd = dt.find_next_sibling("dd")
            if dd:
                details[dt.text.strip()] = dd.text.strip()
        return details

    def parse_listing(self, url: str, soup: BeautifulSoup) -> dict:
        state = self._parse_state(soup)
        details = self._details_grid(soup)
        ld = self._ld_apartment(soup)

        analytics = state.get("analytics") or {}
        coords = state.get("coordinates") or {}
        ld_addr = ld.get("address") or {}
        ld_size = ld.get("floorSize")
        if isinstance(ld_size, list):
            ld_size = ld_size[0] if ld_size else {}
        ld_size = ld_size or {}

        listing_id = self.listing_key(url)
        location_path = analytics.get("locationPath") or ""

        # Floor plan images: Oikotie's own tag on the embedded images list.
        remote_urls, local_paths = [], []
        for img in state.get("images", []):
            tags = img.get("tags") or []
            if not any("floorplan" in str(tag).lower() for tag in tags):
                continue
            src = img.get("url", "")
            if not src:
                continue
            remote_urls.append(src)
            n = len(local_paths)
            saved = self.download_image(src, f"oikotie_{listing_id}_floorplan_{n}.jpg")
            if saved:
                local_paths.append(saved)

        record = {
            "listing_id": listing_id,
            "address": state.get("address", ""),
            "postal_code": analytics.get("zipCode") or ld_addr.get("postalCode", ""),
            "city": location_path.split("/")[0] if location_path else ld_addr.get("addressRegion", ""),
            "district": ld_addr.get("addressLocality", ""),
            "latitude": coords.get("latitude", ""),
            "longitude": coords.get("longitude", ""),
            "build_year": analytics.get("apartmentBuildYear") or details.get("Rakennusvuosi", ""),
            "rooms": ld.get("numberOfRooms", "") or details.get("Huoneita", ""),
            "room_configuration": state.get("roomConfiguration", ""),
            "size_m2": ld_size.get("value", "") or analytics.get("size", ""),
            "floor": details.get("Kerros", ""),
            "building_type": analytics.get("apartmentType") or details.get("Rakennuksen tyyppi", ""),
            "energy_class": details.get("Energialuokka", ""),
            "price_eur": analytics.get("price", ""),
            "sell_status": state.get("sellStatus", ""),
            "status": analytics.get("status", ""),
            "listing_type": analytics.get("listingType", ""),
            "published": analytics.get("published", ""),
            "realtor_company": analytics.get("realtorCompany", ""),
            "realtor_listing_number": analytics.get("realtorListingNumber", ""),
            "floorplan_count": len(remote_urls),
            "floorplan_image_urls": " | ".join(remote_urls),
            "floorplan_local_paths": " | ".join(local_paths),
            "details_json": json.dumps(details, ensure_ascii=False),
        }
        if self.include_agent_names:
            contact = state.get("contact") or {}
            record["agent_name"] = analytics.get("brokerName") or contact.get("contactName", "")
        return record


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Scrape Oikotie for-sale listings + floor plans.")
    ap.add_argument("--limit", type=int, default=15,
                    help="max NEW listings to fetch this run (re-runs resume from listings.jsonl)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests (keep >=2)")
    ap.add_argument("--region", choices=["pirkanmaa", "uusimaa", "both"], default="both",
                    help="which municipality group to keep (default: both)")
    ap.add_argument("--all-finland", action="store_true",
                    help="don't restrict by region at all (uses --shards sitemap shards)")
    ap.add_argument("--shards", type=int, default=None, help="how many sm_ad_*.xml shards to read")
    ap.add_argument("--only-with-floorplan", action="store_true",
                    help="drop listings whose page had no floor plan image")
    ap.add_argument("--include-agent-names", action="store_true",
                    help="keep selling agents' personal names (GDPR: off by default)")
    args = ap.parse_args()

    scraper = OikotieScraper(
        output_dir=args.output_dir,
        request_delay=args.delay,
        cities=None if args.all_finland else REGIONS[args.region],
        sitemap_shards=args.shards,
        include_agent_names=args.include_agent_names,
    )
    df = scraper.run(limit=args.limit, only_with_floorplan=args.only_with_floorplan)
    print(df.head())
    print(f"\n{len(df)} listings, {int(df['floorplan_count'].fillna(0).astype(int).gt(0).sum())} with a floor plan"
          if len(df) else "no listings scraped")