import re
import socket
import ssl
import math
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Tuple

SUSPICIOUS_TLDS = {"tk","ml","ga","cf","gq","top","xyz","club","work","click","fit","buzz","rest","zip","mov"}
BRAND_KEYWORDS = {"paypal","google","apple","microsoft","facebook","instagram","whatsapp","amazon","netflix","bank","secure","login","verify","account","update","wallet","crypto"}
SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","cutt.ly","rebrand.ly","shorturl.at"}
RISK_WORDS = {"login","verify","update","secure","account","confirm","password","signin","banking","wallet","free","bonus","gift","limited","urgent"}

FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens", "num_digits",
    "num_special_chars", "has_ip", "has_at", "has_https", "has_suspicious_tld",
    "is_shortened", "brand_keyword_count", "risk_word_count", "subdomain_count",
    "query_param_count", "entropy", "domain_age_days", "ssl_valid", "reachable",
    "has_forms", "external_link_ratio"
]

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url

def _host_parts(hostname: str) -> Tuple[str, str]:
    hostname = (hostname or "").lower().strip(".")
    parts = hostname.split(".") if hostname else []
    tld = parts[-1] if len(parts) >= 2 else ""
    domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    return domain, tld

def _is_ip(hostname: str) -> int:
    if not hostname:
        return 0
    return int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname)))

def safe_domain_age_days(hostname: str) -> int:
    try:
        import whois
        w = whois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if not created:
            return -1
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - created).days)
    except Exception:
        return -1

def safe_ssl_valid(hostname: str, timeout: float = 2.0) -> int:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                ssock.getpeercert()
        return 1
    except Exception:
        return 0

def safe_behavioral_features(url: str, timeout: float = 3.0) -> Dict[str, float]:
    features = {"reachable": 0, "has_forms": 0, "external_link_ratio": 0.0}
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "URLScope/1.0"})
        features["reachable"] = int(resp.status_code < 500)
        soup = BeautifulSoup(resp.text[:200000], "html.parser")
        features["has_forms"] = int(bool(soup.find_all("form")))
        parsed_host = urllib.parse.urlparse(url).hostname or ""
        links = [a.get("href", "") for a in soup.find_all("a")]
        absolute = [h for h in links if h.startswith(("http://", "https://"))]
        external = [h for h in absolute if (urllib.parse.urlparse(h).hostname or "") != parsed_host]
        features["external_link_ratio"] = round(len(external) / max(1, len(absolute)), 3)
    except Exception:
        pass
    return features

def extract_features(url: str, enable_live_checks: bool = False) -> Dict[str, float]:
    url = _normalize_url(url)
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    domain, tld = _host_parts(hostname)
    full_text = f"{hostname} {parsed.path} {parsed.query}".lower()
    feats: Dict[str, float] = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "num_dots": url.count("."),
        "num_hyphens": hostname.count("-"),
        "num_digits": sum(ch.isdigit() for ch in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "has_ip": _is_ip(hostname),
        "has_at": int("@" in url),
        "has_https": int(parsed.scheme == "https"),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortened": int(domain in SHORTENERS),
        "brand_keyword_count": sum(kw in full_text for kw in BRAND_KEYWORDS),
        "risk_word_count": sum(kw in full_text for kw in RISK_WORDS),
        "subdomain_count": max(0, len(hostname.split(".")) - 2) if hostname else 0,
        "query_param_count": len(urllib.parse.parse_qs(parsed.query)),
        "entropy": round(_entropy(url), 3),
        "domain_age_days": -1,
        "ssl_valid": int(parsed.scheme == "https"),
        "reachable": 0,
        "has_forms": 0,
        "external_link_ratio": 0.0,
    }
    if enable_live_checks and hostname and not feats["has_ip"]:
        feats["domain_age_days"] = safe_domain_age_days(hostname)
        feats["ssl_valid"] = safe_ssl_valid(hostname)
        feats.update(safe_behavioral_features(url))
    return feats

def feature_frame(url: str, enable_live_checks: bool = False):
    import pandas as pd
    feats = extract_features(url, enable_live_checks)
    return pd.DataFrame([[feats[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER), feats
