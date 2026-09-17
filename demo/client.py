"""Small HTTP boundary. The UI never imports models or loads user artifacts."""
from dataclasses import dataclass
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    pass


@dataclass
class Result:
    data: dict
    source: str = ""
    cache: str = ""


def fetch(path, **params):
    base = os.environ.get("RECOMART_API_URL", "http://127.0.0.1:8000").rstrip("/")
    url = base + path + ("?" + urlencode(params) if params else "")
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=8) as response:
            return Result(json.load(response), response.headers.get("X-RecoMart-Source", ""),
                          response.headers.get("X-RecoMart-Cache", ""))
    except HTTPError as exc:
        if exc.code == 404:
            raise APIError("We couldn't find that product. Try another search.") from exc
        if exc.code == 422:
            raise APIError("Check your selection and try again.") from exc
        raise APIError("Recommendations are temporarily unavailable. Please try again.") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise APIError("We can't reach RecoMart right now. Please retry in a moment.") from exc


def parse_user_id(value):
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal() or not 1 <= int(text) <= 9223372036854775807:
        raise ValueError("Enter a shopper ID from 1 to 9,223,372,036,854,775,807.")
    return int(text)


def shopping_list_csv(products):
    import csv
    import io
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Product ID", "Product", "Aisle", "Department"])
    for product in products:
        # Spreadsheet-safe cells even if catalog content starts with a formula.
        values = [str(product[k]) for k in ("product_id", "product_name", "aisle", "department")]
        writer.writerow(["'" + v if v[:1] in "=+-@\t\r\n" else v for v in values])
    return output.getvalue()
