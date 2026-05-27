from flask import Flask, render_template, request, jsonify
import pickle, gzip, re, requests, os
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# ── Load bundle ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_bundle():
    # Try compressed first, then uncompressed
    for filename in ['model_bundle.pkl.gz', 'model_bundle.pkl']:
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            opener = gzip.open if filename.endswith('.gz') else open
            with opener(path, 'rb') as f:
                bundle = pickle.load(f)
            print(f"✅ Loaded bundle: {filename} from {BASE_DIR}")
            return bundle
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    # Nothing loaded — print what IS in the directory to help debug
    print(f"❌ No bundle found in {BASE_DIR}")
    print(f"   Files present: {os.listdir(BASE_DIR)}")
    return None

bundle = load_bundle()

if bundle:
    model              = bundle['model']
    vectorizer         = bundle['vectorizer']
    FEATURE_NAMES      = bundle['feature_names']
    WORD_GROUPS        = bundle['word_groups']
    DISPOSABLE_DOMAINS = bundle['disposable_domains']
    BRAND_DOMAINS      = bundle['brand_domains']
    USE_ML_MODEL       = True
    print("✅ ML model ready")
else:
    model = vectorizer = FEATURE_NAMES = None
    USE_ML_MODEL = False
    print("⚠️  Running in RULE-BASED fallback mode")

    # Hardcoded fallback so app never crashes
    WORD_GROUPS = {
        'urgency':     ['urgent','immediately','asap','expires','act now','hurry',
                        'deadline','today only','right now','do not delay','limited time',
                        'final notice','last chance','expiring','respond now'],
        'action':      ['click','verify','confirm','update','login','validate',
                        'authorize','authenticate','sign in','click here','click below',
                        'follow the link','open attachment','download','submit','access now'],
        'finance':     ['bank','account','credit','debit','password','wire','transfer',
                        'paypal','bitcoin','payment','invoice','billing','wallet',
                        'transaction','funds','refund','tax','irs','stimulus','deposit'],
        'threat':      ['suspended','disabled','locked','blocked','unauthorized',
                        'compromised','hacked','terminated','breach','violation',
                        'closed','restricted','flagged','illegal','criminal','arrested'],
        'prize':       ['winner','prize','free','congratulations','million','inheritance',
                        'lottery','selected','chosen','lucky','reward','claim',
                        'unclaimed','bonus','gift card','won','jackpot'],
        'identity':    ['social security','ssn','date of birth','mother maiden',
                        'security question','pin number','card number','cvv',
                        'full name','home address','passport','drivers license'],
        'impersonate': ['paypal','amazon','apple','microsoft','google','netflix',
                        'bank of america','chase','wells fargo','irs','fbi',
                        'government','official','fedex','ups','dhl','usps'],
        'attachment':  ['attachment','attached','open file','pdf','doc','docx',
                        'spreadsheet','zip','invoice attached','receipt'],
    }
    DISPOSABLE_DOMAINS = {
        'mailinator.com','tempmail.com','guerrillamail.com','throwam.com',
        'yopmail.com','sharklasers.com','maildrop.cc','trashmail.com',
        '10minutemail.com','fakeinbox.com','dispostable.com','spamgourmet.com',
        'getairmail.com','filzmail.com','safetymail.info','spamherelots.com',
    }
    BRAND_DOMAINS = {
        'paypal':'paypal.com','amazon':'amazon.com','apple':'apple.com',
        'microsoft':'microsoft.com','google':'google.com','netflix':'netflix.com',
        'chase':'chase.com','wellsfargo':'wellsfargo.com',
    }

# ── Text cleaning ─────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' IPADDR ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Sender features ───────────────────────────────────────────────
def extract_sender_features(sender_email):
    feat = {
        'sender_is_disposable':0,'sender_domain_mismatch':0,
        'sender_has_numbers':0,'sender_subdomain_count':0,
        'sender_tld_suspicious':0,'sender_brand_spoof':0,
        'sender_long_domain':0,'sender_has_hyphen':0,
    }
    if not sender_email or '@' not in str(sender_email):
        return feat
    domain = str(sender_email).lower().split('@')[-1]
    feat['sender_is_disposable']   = int(domain in DISPOSABLE_DOMAINS)
    feat['sender_has_numbers']     = int(bool(re.search(r'\d', domain)))
    feat['sender_subdomain_count'] = max(len(domain.split('.')) - 2, 0)
    feat['sender_tld_suspicious']  = int(bool(re.search(
        r'\.(ru|tk|xyz|top|click|loan|work|gq|cn|pw)\b', domain)))
    feat['sender_long_domain']     = int(len(domain) > 30)
    feat['sender_has_hyphen']      = int('-' in domain)
    brand_spoof = 0
    for brand, real_domain in BRAND_DOMAINS.items():
        if brand in domain and domain != real_domain:
            brand_spoof = 1
            break
    feat['sender_brand_spoof']     = brand_spoof
    feat['sender_domain_mismatch'] = int(
        any(b in str(sender_email).lower().split('@')[0] for b in BRAND_DOMAINS)
        and brand_spoof == 0
    )
    return feat

# ── Full feature extraction ───────────────────────────────────────
def extract_features(text_raw, sender_email=''):
    raw      = str(text_raw).lower()
    text_str = str(text_raw)
    feat     = {}

    for group_name, words in WORD_GROUPS.items():
        feat[f'{group_name}_count']   = sum(1 for w in words if w in raw)
        feat[f'{group_name}_present'] = int(feat[f'{group_name}_count'] > 0)

    urls = re.findall(r'https?://\S+|www\.\S+', raw)
    feat['has_url']            = int(len(urls) > 0)
    feat['url_count']          = len(urls)
    feat['has_http_only']      = int('http://' in raw and 'https://' not in raw)
    feat['has_ip_url']         = int(bool(re.search(r'https?://\d{1,3}\.\d{1,3}', raw)))
    feat['suspicious_tld']     = int(bool(re.search(
        r'\.(ru|tk|xyz|top|click|loan|work|gq|pw)\b', raw)))
    feat['url_has_at_sign']    = int(any('@' in u for u in urls))
    feat['url_subdomain_deep'] = int(any(u.count('.') > 3 for u in urls))
    feat['url_long']           = int(any(len(u) > 75 for u in urls))

    words_ = text_str.split()
    feat['exclamation_count'] = text_str.count('!')
    feat['question_count']    = text_str.count('?')
    feat['caps_ratio']        = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)
    feat['text_length']       = len(text_str)
    feat['word_count']        = len(words_)
    feat['avg_word_length']   = np.mean([len(w) for w in words_]) if words_ else 0
    feat['unique_word_ratio'] = len(set(words_)) / max(len(words_), 1)
    feat['html_tag_count']    = len(re.findall(r'<[^>]+>', raw))
    feat['has_html']          = int(feat['html_tag_count'] > 0)

    feat['urgency_x_action']  = feat['urgency_count']     * feat['action_count']
    feat['finance_x_threat']  = feat['finance_count']     * feat['threat_count']
    feat['impersonate_x_url'] = feat['impersonate_count'] * feat['has_url']
    feat['threat_x_action']   = feat['threat_count']      * feat['action_count']

    feat.update(extract_sender_features(sender_email))
    return feat

# ── Free email reputation (no API key) ───────────────────────────
def check_hibp_email_free(email):
    result = {
        'checked':False,'breached':False,'breach_count':0,'breach_names':[],
        'reputation':'UNKNOWN','malicious':False,'suspicious':False,
        'spam_flag':False,'disposable':False,'source':None,'error':None,
    }
    try:
        resp = requests.get(
            f"https://emailrep.io/{email}",
            headers={'User-Agent':'PhishGuardAI/1.0'},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            result['checked']    = True
            result['source']     = 'emailrep.io'
            result['reputation'] = data.get('reputation','unknown').upper()
            result['suspicious'] = data.get('suspicious', False)
            result['malicious']  = data.get('suspicious', False)
            details = data.get('details', {})
            result['breached']      = details.get('data_breach', False)
            result['spam_flag']     = details.get('spam', False)
            result['disposable']    = details.get('disposable', False)
            result['free_provider'] = details.get('free_provider', False)
            result['breach_count']  = 1 if result['breached'] else 0
            return result
    except Exception as e:
        result['error'] = f"emailrep failed: {e}"
    try:
        resp = requests.get(
            f"https://leakcheck.io/api/public?check={email}",
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            result['checked']      = True
            result['source']       = 'leakcheck.io'
            result['breached']     = data.get('found', False)
            sources                = data.get('sources', [])
            result['breach_count'] = len(sources)
            result['breach_names'] = [s.get('name','?') for s in sources[:5]]
            return result
    except Exception as e:
        result['error'] = str(result.get('error','')) + f" | leakcheck failed: {e}"
    return result

# ── Sender reputation ─────────────────────────────────────────────
def check_sender_reputation(sender_email):
    if not sender_email or '@' not in str(sender_email):
        return {'risk_label':'UNKNOWN','risk_score':0,'checked':False}
    report = {
        'email':sender_email,'breached':False,'breach_count':0,'breach_names':[],
        'is_disposable':False,'brand_spoof':False,'suspicious_tld':False,
        'spam_flag':False,'risk_score':0,'risk_label':'LOW','checked':True,
    }
    domain = sender_email.lower().split('@')[-1]
    if domain in DISPOSABLE_DOMAINS:
        report['is_disposable'] = True
        report['risk_score']   += 40
    if re.search(r'\.(ru|tk|xyz|top|click|loan|work|gq|cn|pw)$', domain):
        report['suspicious_tld'] = True
        report['risk_score']    += 30
    for brand, real_domain in BRAND_DOMAINS.items():
        if brand in domain and domain != real_domain:
            report['brand_spoof']  = True
            report['spoof_target'] = brand
            report['risk_score']  += 50
            break
    email_rep = check_hibp_email_free(sender_email)
    if email_rep.get('breached'):
        report['breached']     = True
        report['breach_count'] = email_rep.get('breach_count', 1)
        report['breach_names'] = email_rep.get('breach_names', [])
        report['risk_score']  += min(report['breach_count'] * 5, 30)
    if email_rep.get('suspicious') or email_rep.get('malicious'):
        report['risk_score'] += 25
    if email_rep.get('spam_flag'):
        report['spam_flag']   = True
        report['risk_score'] += 20
    if email_rep.get('disposable'):
        report['is_disposable'] = True
        report['risk_score']   += 40
    report['reputation_source'] = email_rep.get('source', 'local-only')
    report['api_reputation']    = email_rep.get('reputation', 'N/A')
    score = report['risk_score']
    report['risk_label'] = 'HIGH' if score >= 60 else ('MEDIUM' if score >= 30 else 'LOW')
    return report

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status':     'ok',
        'ml_model':   USE_ML_MODEL,
        'mode':       'ML' if USE_ML_MODEL else 'RULE-BASED',
        'base_dir':   BASE_DIR,
        'files':      os.listdir(BASE_DIR),
    })

@app.route('/predict', methods=['POST'])
def predict():
    sender  = request.form.get('sender',  '').strip()
    subject = request.form.get('subject', '').strip()
    body    = request.form.get('body',    '').strip()
    url     = request.form.get('url',     '').strip()

    if not body:
        return jsonify({'error': 'Email body is required'})
    if len(body) > 10000:
        return jsonify({'error': 'Input too large (max 10,000 characters)'})

    raw_text = f"{subject} {body} {url}"
    feats    = extract_features(raw_text, sender)

    if USE_ML_MODEL:
        cleaned   = clean_text(raw_text)
        tfidf_vec = vectorizer.transform([cleaned])
        hand_vec  = csr_matrix(
            np.array([feats[f] for f in FEATURE_NAMES]).reshape(1, -1).astype(float)
        )
        X           = hstack([tfidf_vec, hand_vec])
        prediction  = int(model.predict(X)[0])
        probability = model.predict_proba(X)[0]
        phishing_prob = round(float(probability[1]) * 100, 2)
        safe_prob     = round(float(probability[0]) * 100, 2)
    else:
        score  = 0
        score += feats.get('urgency_count',     0) * 15
        score += feats.get('threat_count',      0) * 15
        score += feats.get('finance_count',     0) * 10
        score += feats.get('impersonate_count', 0) * 20
        score += feats.get('prize_count',       0) * 12
        score += feats.get('action_count',      0) * 8
        score += feats.get('has_ip_url',        0) * 30
        score += feats.get('has_http_only',     0) * 15
        score += feats.get('suspicious_tld',    0) * 25
        score += feats.get('sender_brand_spoof',    0) * 40
        score += feats.get('sender_is_disposable',  0) * 30
        score += feats.get('sender_tld_suspicious', 0) * 25
        score += feats.get('urgency_x_action',  0) * 5
        score += feats.get('finance_x_threat',  0) * 5
        if feats.get('caps_ratio', 0) > 0.3:    score += 10
        if feats.get('exclamation_count', 0) > 2: score += 5
        score         = min(score, 100)
        prediction    = 1 if score >= 40 else 0
        phishing_prob = round(float(score), 2)
        safe_prob     = round(100.0 - phishing_prob, 2)

    raw_lower  = raw_text.lower()
    detected   = []
    group_hits = {}
    for group, words in WORD_GROUPS.items():
        hits = [w for w in words if w in raw_lower]
        if hits:
            group_hits[group] = hits
            detected += hits
    detected = list(set(detected))[:12]

    urls_in_body = re.findall(r'http\S+', body)
    all_urls     = ([url] if url else []) + urls_in_body
    risky_url    = any(
        any(p in u.lower() for p in [
            'bit.ly','tinyurl','192.168','.ru/','.tk/','.xyz/',
            'login-','secure-','verify-','-paypal','-amazon','-microsoft'
        ]) for u in all_urls
    )

    threat_signals = []
    if feats.get('urgency_count',     0) > 0: threat_signals.append('Urgency language')
    if feats.get('threat_count',      0) > 0: threat_signals.append('Threat language')
    if feats.get('finance_count',     0) > 0: threat_signals.append('Financial keywords')
    if feats.get('impersonate_count', 0) > 0: threat_signals.append('Brand impersonation')
    if feats.get('prize_count',       0) > 0: threat_signals.append('Prize/reward language')
    if feats.get('attachment_count',  0) > 0: threat_signals.append('Attachment reference')
    if feats.get('has_url',           0):     threat_signals.append('Contains URLs')
    if feats.get('has_http_only',     0):     threat_signals.append('Insecure HTTP link')
    if feats.get('has_ip_url',        0):     threat_signals.append('IP-based URL detected')
    if feats.get('sender_brand_spoof',    0): threat_signals.append('Sender spoofs a brand')
    if feats.get('sender_is_disposable',  0): threat_signals.append('Disposable sender domain')
    if feats.get('sender_tld_suspicious', 0): threat_signals.append('Suspicious sender TLD')
    if feats.get('caps_ratio',        0) > 0.3:  threat_signals.append('Excessive CAPS usage')
    if feats.get('exclamation_count', 0) > 2:     threat_signals.append('Excessive exclamation marks')

    sender_rep = check_sender_reputation(sender) if sender else {
        'risk_label':'N/A','risk_score':0,'checked':False
    }

    return jsonify({
        'prediction':           prediction,
        'label':                'PHISHING' if prediction == 1 else 'SAFE',
        'phishing_probability': phishing_prob,
        'safe_probability':     safe_prob,
        'suspicious_keywords':  detected,
        'keyword_groups':       group_hits,
        'risky_url_detected':   risky_url,
        'threat_signals':       threat_signals,
        'sender_reputation':    sender_rep,
        'model_mode':           'ML' if USE_ML_MODEL else 'RULE-BASED',
        'email_preview':        body[:300] + '...' if len(body) > 300 else body,
    })

if __name__ == '__main__':
    app.run(debug=True)