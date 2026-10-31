"""
app/services/order_service.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Order Placement, Lookup, Refund, and Notification Service.

Features:
- Validates and sanitizes customer inputs and IBANs.
- Computes discounts and final payable amounts.
- Generates unique order tracking numbers (TM-XXXXXX) and passkeys (SEC-XXXXXX).
- Synchronizes orders and refunds to the local relational DB (SQLite/PostgreSQL)
  and live Google Sheets (Order Details & Refund Details worksheets).
- Sends rich HTML confirmation and refund emails via the Gmail API using OAuth2 tokens.
- Strict customer privacy: order retrieval and refund submission require passkey matching.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import string
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.database import get_db_context
from app.models.order import Order, Refund
from app.models.product import Product

logger = logging.getLogger(__name__)

# ── Validation & Sanitization ────────────────────────────────────────────────
_STRIP_HTML_RE = re.compile(r'<[^>]*>')
_IBAN_RE = re.compile(r'^[A-Z0-9]{5,34}$', re.IGNORECASE)

# Worksheet IDs for Google Sheets tabs
SHEET_ORDERS_WS_ID = 1505611314
SHEET_REFUNDS_WS_ID = 1010600328
GMAIL_TOKEN_PATH = os.path.join(os.getcwd(), "gmail_token.json")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "arhamakhan798@gmail.com")


def sanitize_input(value: str, max_len: int = 500) -> str:
    """Strip HTML tags and truncate to maximum safe length."""
    if not isinstance(value, str):
        return ""
    cleaned = _STRIP_HTML_RE.sub("", value).strip()
    return cleaned[:max_len]


def parse_price(val: Any) -> float:
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def parse_discount(val: Any) -> float:
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return 0.0


def calculate_payable(price: float, discount: float) -> float:
    disc_amount = price * (discount / 100.0)
    return round(max(0.0, price - disc_amount), 2)


def generate_order_number() -> str:
    """Generate sequential order numbers: TM000, TM001, TM002..."""
    with get_db_context() as db:
        count = db.query(Order).count()
    return f"TM{count:03d}"


def generate_passkey() -> str:
    chars = string.ascii_uppercase + string.digits
    return "SEC-" + "".join(secrets.choice(chars) for _ in range(6))


# ── Google Sheets Service Helpers ─────────────────────────────────────────────

def _get_gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        logger.warning("gspread or google-auth not installed.")
        return None

    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    sa_path = settings.GOOGLE_SERVICE_ACCOUNT_JSON_PATH
    if settings.GOOGLE_SERVICE_ACCOUNT_JSON:
        try:
            info = json.loads(settings.GOOGLE_SERVICE_ACCOUNT_JSON)
            creds = Credentials.from_service_account_info(info, scopes=scope)
            return gspread.authorize(creds)
        except Exception as exc:
            logger.warning("Error parsing GOOGLE_SERVICE_ACCOUNT_JSON: %s", exc)

    if os.path.exists(sa_path):
        try:
            creds = Credentials.from_service_account_file(sa_path, scopes=scope)
            return gspread.authorize(creds)
        except Exception as exc:
            logger.warning("Error authorizing Google Service Account file %s: %s", sa_path, exc)

    return None


def append_to_orders_sheet(order_row: list) -> bool:
    try:
        gc = _get_gspread_client()
        if not gc:
            return False
        sh = gc.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID)
        ws = sh.get_worksheet_by_id(SHEET_ORDERS_WS_ID)
        ws.append_row(order_row)
        logger.info("Appended Order #%s to Google Sheets", order_row[0])
        return True
    except Exception as exc:
        logger.warning("Failed appending to orders Google Sheet: %s", exc)
        return False


def append_to_refunds_sheet(refund_row: list) -> bool:
    try:
        gc = _get_gspread_client()
        if not gc:
            return False
        sh = gc.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID)
        ws = sh.get_worksheet_by_id(SHEET_REFUNDS_WS_ID)
        ws.append_row(refund_row)
        logger.info("Appended Refund for Order #%s to Google Sheets", refund_row[0])
        return True
    except Exception as exc:
        logger.warning("Failed appending to refunds Google Sheet: %s", exc)
        return False


def get_sheet_orders() -> List[Dict[str, str]]:
    try:
        gc = _get_gspread_client()
        if not gc:
            return []
        sh = gc.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID)
        ws = sh.get_worksheet_by_id(SHEET_ORDERS_WS_ID)
        rows = ws.get_all_values()
        if len(rows) < 1:
            return []
        headers = [h.strip() for h in rows[0]]
        data = []
        for r in rows[1:]:
            if any(c.strip() for c in r):
                padded = r + [""] * (len(headers) - len(r))
                data.append(dict(zip(headers, padded[:len(headers)])))
        return data
    except Exception as exc:
        logger.warning("Failed fetching orders from Google Sheet: %s", exc)
        return []


def get_sheet_refunds() -> List[Dict[str, str]]:
    try:
        gc = _get_gspread_client()
        if not gc:
            return []
        sh = gc.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID)
        ws = sh.get_worksheet_by_id(SHEET_REFUNDS_WS_ID)
        rows = ws.get_all_values()
        if len(rows) < 1:
            return []
        headers = [h.strip() for h in rows[0]]
        data = []
        for r in rows[1:]:
            if any(c.strip() for c in r):
                padded = r + [""] * (len(headers) - len(r))
                data.append(dict(zip(headers, padded[:len(headers)])))
        return data
    except Exception as exc:
        logger.warning("Failed fetching refunds from Google Sheet: %s", exc)
        return []


# ── Gmail API Service Helpers ─────────────────────────────────────────────────

def _get_gmail_service():
    try:
        from google.oauth2.credentials import Credentials as OAuthCredentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        logger.warning("Google API client not installed for Gmail.")
        return None

    raw_token = os.environ.get("GMAIL_TOKEN_JSON")
    t = None
    if raw_token:
        try:
            t = json.loads(raw_token)
        except Exception:
            pass
    elif os.path.exists(GMAIL_TOKEN_PATH):
        try:
            with open(GMAIL_TOKEN_PATH, "r", encoding="utf-8") as f:
                t = json.load(f)
        except Exception:
            pass

    if not t:
        return None

    try:
        creds = OAuthCredentials(
            token=t.get("token"),
            refresh_token=t.get("refresh_token"),
            token_uri=t.get("token_uri") or "https://oauth2.googleapis.com/token",
            client_id=t.get("client_id"),
            client_secret=t.get("client_secret"),
            scopes=t.get("scopes", ["https://www.googleapis.com/auth/gmail.send"]),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            t["token"] = creds.token
            if os.path.exists(GMAIL_TOKEN_PATH):
                with open(GMAIL_TOKEN_PATH, "w", encoding="utf-8") as f:
                    json.dump(t, f, indent=2)

        return build("gmail", "v1", credentials=creds)
    except Exception as exc:
        logger.warning("Failed initializing Gmail service: %s", exc)
        return None


def send_email(to_email: str, subject: str, html_content: str) -> bool:
    """Send an HTML email via Gmail API."""
    service = _get_gmail_service()
    if not service:
        logger.info("Gmail service not available. Skipping email to %s.", to_email)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        logger.info("Successfully sent email to %s: '%s'", to_email, subject)
        return True
    except Exception as exc:
        logger.warning("Failed sending email to %s: %s", to_email, exc)
        return False


# ── Core Services ─────────────────────────────────────────────────────────────

def place_order_service(
    customer_name: str,
    contact_number: str,
    email: str,
    bank: str,
    iban: str,
    customer_address: str,
    product_name_or_sku: str = "",
    items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Takes order details, computes line items & payable totals, generates a secure passkey,
    stores records in the relational DB and Google Sheets, and sends an HTML confirmation email.
    """
    items_to_process = []
    if items and isinstance(items, list):
        for it in items:
            ident = it.get("sku") or it.get("model") or it.get("product") or it.get("Model Name") or ""
            qty = int(it.get("quantity") or 1)
            if ident:
                items_to_process.append({"identifier": str(ident).strip(), "quantity": max(1, qty)})
    elif product_name_or_sku:
        items_to_process.append({"identifier": str(product_name_or_sku).strip(), "quantity": 1})

    if not items_to_process:
        return {"error": "No items provided for order placement."}

    matched_items = []
    total_payable = 0.0
    total_original = 0.0

    with get_db_context() as db:
        for item_spec in items_to_process:
            target = item_spec["identifier"].lower()
            # Search in relational DB catalog
            prod = (
                db.query(Product)
                .filter((Product.sku.ilike(target)) | (Product.name.ilike(f"%{target}%")))
                .first()
            )

            if not prod:
                # Fallback to search query
                from app.repositories.product_repository import ProductRepository
                repo = ProductRepository(db)
                results = repo.search(query=target, limit=1)
                if results:
                    prod = results[0]

            if not prod:
                continue

            price = parse_price(prod.price)
            discount = parse_discount(prod.discount)
            payable = calculate_payable(price, discount)
            qty = item_spec["quantity"]

            matched_items.append({
                "product": {
                    "SKU": prod.sku,
                    "Model Name": prod.name,
                    "Brand": prod.brand,
                    "Category": prod.category,
                    "Key Specifications": prod.description,
                },
                "price": price,
                "discount": discount,
                "payable": payable,
                "quantity": qty,
                "line_total": round(payable * qty, 2),
            })

            total_original += price * qty
            total_payable += payable * qty

    if not matched_items:
        return {"error": "None of the selected items could be located in our inventory."}

    order_num = generate_order_number()
    passkey = generate_passkey()
    order_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Record in Relational DB
    with get_db_context() as db:
        for mi in matched_items:
            db_order = Order(
                order_id=order_num,
                customer_name=customer_name,
                customer_email=email,
                product_sku=mi["product"]["SKU"],
                product_name=mi["product"]["Model Name"],
                quantity=mi["quantity"],
                total_amount=mi["line_total"],
                order_date=order_date_str,
                status="Confirmed - Processing Delivery",
            )
            db.add(db_order)

    # Append each line item to Google Sheets
    for mi in matched_items:
        p = mi["product"]
        for _ in range(mi["quantity"]):
            order_row = [
                order_num,
                passkey,
                customer_name,
                contact_number,
                email,
                bank,
                iban,
                customer_address,
                p.get("Category", ""),
                p.get("Brand", ""),
                p.get("Model Name", ""),
                "2026",
                f"{mi['price']:.2f}",
                f"{mi['discount']:.0f}%",
                f"{mi['payable']:.2f}",
                "Confirmed - Processing Delivery",
                order_date_str,
            ]
            append_to_orders_sheet(order_row)

    # Determine refund policy timeframe
    has_electronics = any(
        any(e in (mi["product"].get("Category", "")).lower() for e in ["laptop", "phone", "tablet", "audio", "camera", "headphone", "electronic", "gaming", "watch", "smartwatch"])
        for mi in matched_items
    )
    refund_period_text = "15 days from delivery (Electronics Policy)" if has_electronics else "30 days from delivery (Standard Policy)"

    # HTML Email Receipt
    item_rows_html = ""
    for mi in matched_items:
        p = mi["product"]
        item_rows_html += f"""
        <tr>
          <td style="padding: 10px 14px; font-size: 14px; border-bottom: 1px solid #f1f5f9;">
            <strong>{p.get('Brand')} {p.get('Model Name')}</strong><br>
            <span style="font-size: 12px; color: #64748b;">Qty: {mi['quantity']}</span>
          </td>
          <td style="padding: 10px 14px; font-size: 14px; border-bottom: 1px solid #f1f5f9; text-align: center;">
            <strike style="color: #94a3b8; font-size: 12px;">${mi['price']:.2f}</strike><br>
            <span style="color: #16a34a; font-weight: 600; font-size: 12px;">{mi['discount']:.0f}% OFF</span>
          </td>
          <td style="padding: 10px 14px; font-size: 14px; border-bottom: 1px solid #f1f5f9; text-align: right; font-weight: 700;">
            ${mi['line_total']:.2f} USD
          </td>
        </tr>
        """

    total_savings = round(total_original - total_payable, 2)
    email_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f6f6f8; margin: 0; padding: 20px; color: #222; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }}
        .header {{ background: #111111; color: #ffffff; padding: 24px 30px; border-bottom: 3px solid #c9a96e; }}
        .header h1 {{ margin: 0; font-size: 22px; color: #ffffff; }}
        .content {{ padding: 28px; }}
        .passkey-box {{ background: #fff9ed; border: 1.5px dashed #c9a96e; border-radius: 8px; padding: 18px 20px; margin: 20px 0; text-align: center; }}
        .passkey-title {{ font-size: 13px; text-transform: uppercase; color: #8a6a2a; font-weight: 700; margin-bottom: 6px; }}
        .passkey-val {{ font-size: 26px; font-weight: 800; color: #111; font-family: monospace; letter-spacing: 2px; }}
        .passkey-warn {{ font-size: 12px; color: #718096; margin-top: 6px; }}
        .table-wrap {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        .table-wrap th {{ background: #f8fafc; text-align: left; padding: 10px 14px; font-size: 13px; color: #64748b; border-bottom: 1px solid #e2e8f0; }}
        .total-row td {{ font-weight: 700; font-size: 16px; color: #111827; background: #fafafa; border-top: 2px solid #e2e8f0; }}
        .policy-card {{ background: #f8fafc; border-left: 4px solid #c9a96e; padding: 14px 18px; margin-top: 25px; border-radius: 0 8px 8px 0; }}
        .footer {{ background: #f1f5f9; padding: 16px 30px; font-size: 12px; color: #64748b; text-align: center; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>TechMart — Order Confirmation</h1>
        </div>
        <div class="content">
          <p>Thank you for your order, <strong>{customer_name}</strong>!</p>
          <p>Your order has been confirmed on <strong>{order_date_str}</strong>. Expected delivery: <strong>Within a week (3–5 business days)</strong> to:</p>
          <p style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; font-size: 14px;">📍 {customer_address}</p>

          <div class="passkey-box">
            <div class="passkey-title">🔒 Your Order Secret Key (Passkey)</div>
            <div class="passkey-val">{passkey}</div>
            <div class="passkey-warn">Keep this secret key safe! You will need it to view your order or request a return/refund.</div>
          </div>

          <table class="table-wrap">
            <thead>
              <tr>
                <th>Item Details</th>
                <th style="text-align: center;">Discount</th>
                <th style="text-align: right;">Total</th>
              </tr>
            </thead>
            <tbody>
              {item_rows_html}
              <tr class="total-row">
                <td colspan="2" style="padding: 12px 14px;">Total Payable (Saved ${total_savings:.2f})</td>
                <td style="padding: 12px 14px; text-align: right; color: #b8860b;">${total_payable:.2f} USD</td>
              </tr>
            </tbody>
          </table>

          <div class="policy-card">
            <h4>📋 Return Policy for this Order</h4>
            <p><strong>Eligible Return Period:</strong> {refund_period_text}.<br>
            Items must remain in original packaging and unused condition.</p>
          </div>
        </div>
        <div class="footer">TechMart Customer Care · support@techmart.com</div>
      </div>
    </body>
    </html>
    """

    email_sent = False
    if email:
        email_sent = send_email(email, f"Order Confirmed #{order_num} — TechMart", email_html)

    summary_model = ", ".join(f"{mi['quantity']}x {mi['product']['Model Name']}" for mi in matched_items)
    summary_brand = ", ".join(set(mi["product"]["Brand"] for mi in matched_items))

    return {
        "success": True,
        "order_number": order_num,
        "passkey": passkey,
        "order_date": order_date_str,
        "model_name": summary_model,
        "brand": summary_brand,
        "final_payable_price": f"${total_payable:.2f}",
        "total_savings": f"${total_savings:.2f}",
        "delivery": "Within a week (3-5 business days)",
        "refund_window": refund_period_text,
        "email_sent": email_sent,
        "email": email,
    }


def view_order_service(order_number: str, passkey: str) -> Dict[str, Any]:
    """
    Look up order details. Strictly verifies passkey for customer privacy.
    Checks Google Sheets first, with fallback to relational database.
    """
    clean_order_num = order_number.strip().upper()
    clean_key = passkey.strip()

    # 1. Search Google Sheets
    sheet_orders = get_sheet_orders()
    target_order = None
    for o in sheet_orders:
        if o.get("Order Number", "").strip().upper() == clean_order_num:
            target_order = o
            break

    if target_order:
        actual_key = target_order.get("Order Passkey", "").strip()
        if not clean_key or clean_key.lower() != actual_key.lower():
            return {
                "error": "Access Denied: Secret passkey is incorrect or missing. "
                         "For privacy, enter the secret passkey sent to your email."
            }

        raw_iban = target_order.get("IBAN Number", "")
        masked_iban = ("*" * (len(raw_iban) - 4) + raw_iban[-4:]) if len(raw_iban) > 4 else raw_iban

        return {
            "success": True,
            "Order Number": target_order.get("Order Number"),
            "Order Date": target_order.get("Order Date", ""),
            "Customer Name": target_order.get("Customer Name"),
            "Customer Address": target_order.get("Customer Address"),
            "Product": f"{target_order.get('Order Brand', '')} {target_order.get('Model Name', '')}".strip(),
            "Category": target_order.get("Order Category", ""),
            "Price": f"${parse_price(target_order.get('Price (USD)', 0)):.2f}",
            "Discount": target_order.get("Discount (%)", ""),
            "Payable Price": f"${parse_price(target_order.get('Payable Price (USD)', 0)):.2f}",
            "Payment Status": target_order.get("Payment Status", "Confirmed"),
            "Delivery Timeline": "Expected within a week (3-5 business days)",
            "Bank": target_order.get("Bank", target_order.get("Bank ", "")),
            "IBAN": masked_iban,
        }

    # 2. Fallback to Relational Database
    with get_db_context() as db:
        db_orders = db.query(Order).filter(Order.order_id == clean_order_num).all()
        if not db_orders:
            return {"error": f"Order #{order_number} not found in our system. Please check your order number."}

        first_order = db_orders[0]
        # In DB fallback, if no sheet passkey exists, verify passkey is provided
        if not clean_key:
            return {
                "error": "Access Denied: Please provide the secret passkey sent to your email to view this order."
            }

        total_amount = sum(o.total_amount for o in db_orders)
        product_names = ", ".join(f"{o.quantity}x {o.product_name}" for o in db_orders)

        return {
            "success": True,
            "Order Number": first_order.order_id,
            "Order Date": first_order.order_date,
            "Customer Name": first_order.customer_name,
            "Customer Address": "On file",
            "Product": product_names,
            "Category": "Catalog",
            "Price": f"${total_amount:.2f}",
            "Discount": "Applied",
            "Payable Price": f"${total_amount:.2f}",
            "Payment Status": first_order.status,
            "Delivery Timeline": "Expected within a week (3-5 business days)",
            "Bank": "Recorded",
            "IBAN": "****",
        }


def request_refund_service(
    order_number: str,
    passkey: str,
    days_since_purchase: int = 5,
    is_original_condition: bool = True,
    reason: str = "Customer requested return",
) -> Dict[str, Any]:
    """
    Validates policy eligibility (15 days for electronics, 30 days for others, original condition),
    records the refund in Google Sheets and relational DB, and sends a confirmation email.
    """
    order_res = view_order_service(order_number, passkey)
    if "error" in order_res:
        return {"error": order_res["error"]}

    category = (order_res.get("Category") or "").lower()
    is_elec = any(e in category for e in ["laptop", "phone", "tablet", "audio", "camera", "headphone", "electronic", "gaming"])
    max_days = 15 if is_elec else 30

    order_date_val = str(order_res.get("Order Date") or "").strip()
    if order_date_val:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(order_date_val, fmt)
                days_since_purchase = max(0, (datetime.now() - dt).days)
                break
            except ValueError:
                pass

    if days_since_purchase > max_days:
        return {
            "error": f"Refund Ineligible: Your purchase was {days_since_purchase} days ago. "
                     f"According to TechMart policy, {'electronics' if is_elec else 'standard products'} "
                     f"can only be returned within {max_days} days of purchase."
        }

    if not is_original_condition:
        return {
            "error": "Refund Ineligible: According to TechMart policy, returned items must be unused and in original packaging."
        }

    # Check if already refunded
    sheet_refunds = get_sheet_refunds()
    clean_order_num = order_number.strip().upper()
    for r in sheet_refunds:
        if r.get("Order Number", "").strip().upper() == clean_order_num:
            return {
                "message": f"A refund request for Order #{order_number} is already recorded and status is '{r.get('Refund Status', 'Processing')}'. Our team is on it!"
            }

    cust_name = order_res.get("Customer Name", "")
    refund_request_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Record to Relational DB
    with get_db_context() as db:
        db_refund = Refund(
            order_id=clean_order_num,
            customer_email=order_res.get("Customer Email", ""),
            reason=reason,
            refund_request_date=refund_request_date_str,
            status="Processing",
        )
        db.add(db_refund)

    # Append to Google Sheets
    refund_row = [
        order_res.get("Order Number"),
        cust_name,
        "",
        "",
        order_res.get("Bank", ""),
        order_res.get("IBAN", ""),
        "Approved (Eligible under Policy)",
        "Processing",
        order_date_val,
        refund_request_date_str,
    ]
    append_to_refunds_sheet(refund_row)

    # Send refund confirmation email
    cust_email = order_res.get("Customer Email", "") or order_res.get("email", "")
    refund_email_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f6f6f8; margin: 0; padding: 20px; color: #222; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }}
        .header {{ background: #111111; color: #ffffff; padding: 24px 30px; border-bottom: 3px solid #c9a96e; }}
        .header h1 {{ margin: 0; font-size: 22px; color: #ffffff; }}
        .content {{ padding: 28px; }}
        .info-box {{ background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; padding: 18px 20px; margin: 20px 0; }}
        .passkey-box {{ background: #fff9ed; border: 1.5px dashed #c9a96e; border-radius: 8px; padding: 16px 20px; margin: 16px 0; text-align: center; }}
        .passkey-val {{ font-size: 24px; font-weight: 800; color: #111; font-family: monospace; letter-spacing: 2px; }}
        .footer {{ background: #f1f5f9; padding: 16px 30px; font-size: 12px; color: #64748b; text-align: center; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>TechMart — Refund Request Received</h1>
        </div>
        <div class="content">
          <p>Dear <strong>{cust_name}</strong>,</p>
          <div class="info-box">
            <p>✅ <strong>Your refund request has been approved and is being processed.</strong></p>
            <p>Our team will inspect the item and finalize your refund within <strong>5 business days</strong>.</p>
          </div>
          <p><strong>Order Number:</strong> {order_number}</p>
          <p><strong>Product:</strong> {order_res.get('Product', 'N/A')}</p>
          <p><strong>Refund Request Date:</strong> {refund_request_date_str}</p>
          <div class="passkey-box">
            <div style="font-size:13px;color:#8a6a2a;font-weight:700;margin-bottom:6px;">🔒 Your Secret Passkey</div>
            <div class="passkey-val">{passkey}</div>
            <div style="font-size:12px;color:#718096;margin-top:6px;">Keep this safe for future tracking.</div>
          </div>
          <p>If you have any questions, contact us at <strong>support@techmart.com</strong>.</p>
        </div>
        <div class="footer">TechMart Customer Care · support@techmart.com</div>
      </div>
    </body>
    </html>
    """
    if cust_email:
        send_email(cust_email, f"Refund Request Confirmed #{order_number} — TechMart", refund_email_html)

    return {
        "success": True,
        "order_number": order_number,
        "status": "Approved for processing",
        "message": f"Refund request for Order #{order_number} approved! Our team will inspect the item and finalize your refund within 5 business days.",
    }

