import argparse
import os
import smtplib
import ssl
import time
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from tempfile import TemporaryDirectory


RECIPIENT = "837655230@qq.com"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser("Send project log files by email.")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs"), help="Directory containing log files.")
    parser.add_argument("--once", action="store_true", help="Send logs once and exit.")
    parser.add_argument("--schedule", action="store_true", help="Wait interval hours, send logs, then repeat.")
    parser.add_argument("--interval-hours", default=24.0, type=float, help="Schedule interval in hours.")
    parser.add_argument("--dry-run", action="store_true", help="Create the attachment and print what would be sent.")
    parser.add_argument("--smtp-host", default=os.environ.get("SMTP_HOST", "smtp.qq.com"))
    parser.add_argument("--smtp-port", default=int(os.environ.get("SMTP_PORT", "465")), type=int)
    parser.add_argument("--smtp-user", default=os.environ.get("SMTP_USER"))
    parser.add_argument("--smtp-password-env", default="SMTP_PASSWORD")
    parser.add_argument("--smtp-from", default=os.environ.get("SMTP_FROM"))
    parser.add_argument("--smtp-ssl", action=argparse.BooleanOptionalAction, default=os.environ.get("SMTP_SSL", "1") != "0")
    return parser.parse_args()


def collect_log_files(log_dir):
    root = Path(log_dir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in log directory: {root}")
    return root, files


def build_log_zip(log_dir, output_dir):
    root, files = collect_log_files(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = Path(output_dir) / f"logs_{timestamp}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root.parent))
    return archive_path, files


def build_message(sender, archive_path, files):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = RECIPIENT
    message["Subject"] = f"DINO-SAM-GECO2 training logs {datetime.now():%Y-%m-%d %H:%M:%S}"
    file_list = "\n".join(f"- {path}" for path in files)
    message.set_content(
        "Attached is the zipped log directory from DINO-SAM-GECO2.\n\n"
        f"Recipient: {RECIPIENT}\n"
        f"Files:\n{file_list}\n"
    )
    with open(archive_path, "rb") as attachment:
        message.add_attachment(
            attachment.read(),
            maintype="application",
            subtype="zip",
            filename=archive_path.name,
        )
    return message


def send_email(args, archive_path, files):
    sender = args.smtp_from or args.smtp_user
    if not sender:
        raise ValueError("SMTP sender is required. Set SMTP_USER or SMTP_FROM.")
    if not args.smtp_user:
        raise ValueError("SMTP username is required. Set SMTP_USER.")

    password = os.environ.get(args.smtp_password_env)
    if not password:
        raise ValueError(f"SMTP password is required. Set environment variable {args.smtp_password_env}.")

    message = build_message(sender, archive_path, files)
    if args.smtp_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, context=context) as smtp:
            smtp.login(args.smtp_user, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(args.smtp_host, args.smtp_port) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(args.smtp_user, password)
            smtp.send_message(message)


def send_logs_once(args):
    with TemporaryDirectory() as temp_dir:
        archive_path, files = build_log_zip(args.log_dir, temp_dir)
        if args.dry_run:
            print(f"Dry run: would send {archive_path.name} to {RECIPIENT}")
            print(f"Log files: {len(files)}")
            for path in files:
                print(path)
            return

        send_email(args, archive_path, files)
        print(f"Sent {archive_path.name} to {RECIPIENT}")


def run_schedule(args):
    interval_seconds = max(args.interval_hours, 0.0) * 3600
    while True:
        print(f"Waiting {args.interval_hours} hour(s) before sending logs to {RECIPIENT}.")
        time.sleep(interval_seconds)
        send_logs_once(args)


def main():
    args = parse_args()
    if args.schedule:
        run_schedule(args)
    else:
        send_logs_once(args)


if __name__ == "__main__":
    main()
