# models/train.py
import os
import sys

# Make sure we can import from the models folder
sys.path.append(os.path.dirname(__file__))

from models.Email_Classifier import EmailPhishingClassifier
from models.Url_Classifier  import URLThreatClassifier

# ── Create the models folder if it doesn't exist ──
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# ══════════════════════════════════════════
# TRAIN EMAIL CLASSIFIER
# ══════════════════════════════════════════
print("Training email classifier...")

email_samples = [
    # Phishing emails (label = 1)
    ("Your account has been suspended. Verify immediately.", 1),
    ("Dear customer, click here to confirm your bank account login.", 1),
    ("URGENT: Your PayPal account needs verification. Login now!", 1),
    ("Congratulations! You won $1,000,000. Claim your prize now.", 1),
    ("Your password will expire. Update it immediately here.", 1),
    ("Security alert: Unusual login detected. Verify your identity.", 1),
    ("Your account will be closed in 24 hours unless you confirm.", 1),
    ("Click this link to reset your password or lose access forever.", 1),
    ("Dear user, your Apple ID has been locked. Verify now.", 1),
    ("Your Netflix subscription failed. Update payment details here.", 1),
    ("You have a pending tax refund. Submit your details to claim it.", 1),
    ("Your DHL package is on hold. Pay the customs fee now.", 1),
    ("Microsoft Security Alert: Sign in detected. Confirm identity.", 1),
    ("Dear valued customer, your bank account needs urgent attention.", 1),
    ("Win a free iPhone! Click here to claim your reward.", 1),
    ("Final warning: Your email will be deactivated. Confirm now.", 1),
    ("Your Amazon order is on hold due to payment issues. Verify now.", 1),
    ("Suspicious activity on your account. Login to secure it now.", 1),

    # Legitimate emails (label = 0)
    ("Hi team, the sprint planning meeting is at 2pm tomorrow.", 0),
    ("Please find attached the invoice for Q3 services.", 0),
    ("Meeting notes from yesterday's standup are in the shared doc.", 0),
    ("The pull request has been merged into the main branch.", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("The report you requested is attached. Let me know if you need changes.", 0),
    ("Reminder: Company all-hands meeting is on Friday at 3pm.", 0),
    ("Your order has been shipped. Expected delivery: Monday.", 0),
    ("Thank you for your purchase. Your receipt is attached.", 0),
    ("The new feature release is scheduled for next Wednesday.", 0),
    ("Can we reschedule our 1:1 to Thursday afternoon?", 0),
    ("Please review the attached document before our meeting.", 0),
    ("The database backup completed successfully at 3am.", 0),
    ("Your subscription has been renewed. Thank you for staying with us.", 0),
    ("Here are the action items from today's client call.", 0),
    ("The quarterly performance review is due by end of month.", 0),
    ("Your flight booking confirmation is attached.", 0),
    ("Welcome to the team! Here is your onboarding schedule.", 0),
]

texts  = [s[0] for s in email_samples]
labels = [s[1] for s in email_samples]

email_clf = EmailPhishingClassifier()
email_clf.train(texts, labels)

# Save into the models/ folder
save_path = os.path.join(os.path.dirname(__file__), "email_model.pkl")
email_clf.save(save_path)
print(f"Email model saved to: {save_path}")


# ══════════════════════════════════════════
# TRAIN URL CLASSIFIER
# ══════════════════════════════════════════
print("\nTraining URL classifier...")

url_samples = [
    # Malicious URLs (label = 1)
    ("http://paypa1-secure-login.xyz/verify-account", 1),
    ("http://192.168.1.1/admin/steal-credentials.php", 1),
    ("https://amazon-prize-winner.free-server.com/claim", 1),
    ("http://secure-bankofamerica.phishing-site.ru/login", 1),
    ("http://update-your-netflix.xyz/payment-details", 1),
    ("http://apple-id-locked.verify-now.tk/confirm", 1),
    ("http://www.paypal.account-suspended.pw/login", 1),
    ("http://free-iphone-winner.click-here.xyz/claim", 1),
    ("http://203.0.113.42/steal/credentials.php", 1),
    ("http://microsofft-security-alert.xyz/signin", 1),
    ("http://login.bank.secure-update.ru/verify", 1),
    ("https://tax-refund-gov.phishing.xyz/submit", 1),

    # Safe URLs (label = 0)
    ("https://google.com/search?q=python+tutorial", 0),
    ("https://github.com/user/repository", 0),
    ("https://stackoverflow.com/questions/12345/how-to-use-python", 0),
    ("https://www.amazon.com/dp/B08N5WRWNW", 0),
    ("https://mail.google.com/mail/u/0/", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    ("https://docs.python.org/3/library/os.html", 0),
    ("https://en.wikipedia.org/wiki/Phishing", 0),
    ("https://www.microsoft.com/en-us/microsoft-365", 0),
    ("https://linkedin.com/in/yourprofile", 0),
    ("https://portal.azure.com/#home", 0),
    ("https://npmjs.com/package/express", 0),
]

urls       = [s[0] for s in url_samples]
url_labels = [s[1] for s in url_samples]

url_clf = URLThreatClassifier()
url_clf.train(urls, url_labels)

save_path_url = os.path.join(os.path.dirname(__file__), "url_model.pkl")
url_clf.save(save_path_url)
print(f"URL model saved to: {save_path_url}")

print("\nAll models trained and saved successfully!")
print("You can now run: uvicorn main:app --reload")