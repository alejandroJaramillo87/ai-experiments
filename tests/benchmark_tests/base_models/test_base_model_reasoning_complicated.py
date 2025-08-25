import requests
import json
import time
import os
import textwrap

# Configuration for base model completions
API_URL = "http://127.0.0.1:8001/v1/completions"  # Completions endpoint, not chat
HEADERS = {
    "Content-Type": "application/json"
}
os.makedirs("test_results", exist_ok=True)

# ======================================================================================
# ADVANCED LONG-CONTEXT REASONING TEST SUITE
# ======================================================================================
# This suite is designed to rigorously evaluate a model's long-context reasoning
# capabilities, requiring it to process, synthesize, and recall information from
# thousands of tokens of preceding text.
# ======================================================================================
COMPLEX_TEST_CASES = [
    {
        "name": "Complex Test 1: Multi-Document Synthesis from Scientific Abstracts",
        "prompt": """
CONTEXT:
Below are three fictional research paper abstracts concerning "Chroniton Particles," a newly discovered form of matter.

Document 1: "Temporal Dilation Effects of High-Frequency Chroniton Particle Streams"
Author: Dr. Aris Thorne, Cambridge University
Publication: Journal of Quantum Physics, Vol. 42, Issue 3
Abstract: Our research investigates the temporal effects of high-frequency ( >1 THz) Chroniton Particle (CP) streams on localized spacetime. We utilized a Mark-IV Chroniton Emitter to generate a stable CP beam, directed at a cesium atomic clock array. The experiment, conducted in a vacuum chamber shielded from external gravitational and magnetic fields, revealed a consistent temporal dilation effect. The clocks within the beam's path experienced time at a rate 0.003% slower than control clocks. This dilation scales linearly with the frequency of the CP stream. However, a significant energy decay was observed in the particles post-interaction, suggesting a direct transfer of energy is responsible for the dilation, rather than a gravitational mimicry effect as previously hypothesized by Chen et al. The energy decay manifests as a cascade of low-energy tachyons, a phenomenon not predicted by current models. This discovery opens new avenues for exploring controllable temporal fields but also highlights the need for a revised theoretical framework for CP interactions.

Document 2: "Biochemical Interactions of Low-Energy Chroniton Fields"
Author: Dr. Lena Petrova, MIT
Publication: Nature Biotechnology, Vol. 18, Issue 9
Abstract: This study explores the effects of low-energy (<100 GHz) Chroniton Particle (CP) fields on organic molecules. We exposed various protein samples, including hemoglobin and insulin, to a sustained low-energy CP field for 72 hours. Mass spectrometry and circular dichroism spectroscopy revealed a remarkable 15% increase in protein structural stability and a significant reduction in denaturation rates when exposed to heat stress. The mechanism appears to involve CP-mediated reinforcement of hydrogen bonds within the protein's tertiary structure. This effect is temporary, with protein stability reverting to baseline approximately 24 hours after removal from the field. No tachyon cascade was detected, in stark contrast to the high-frequency experiments conducted by Thorne's team. We hypothesize that at low energy levels, CPs interact non-destructively at a quantum level, temporarily 'lending' stability to molecular structures. These findings have profound implications for pharmacology, potentially enabling the development of drugs with significantly longer shelf lives and enhanced efficacy.

Document 3: "A Unified Field Theory for Chroniton Particle Interactions"
Author: Dr. Jian Li, Stanford University
Publication: Physics Review Letters, Vol. 112, Issue 1
Abstract: Current models fail to reconcile the divergent experimental results of Chroniton Particle (CP) research, specifically the temporal dilation at high frequencies (Thorne) and the biochemical stabilization at low frequencies (Petrova). We propose a Unified Chroniton Field (UCF) theory, which posits that CPs are excitations of a new fundamental field that interacts differently based on its energy density. According to our model, at high energy densities, the UCF field couples with the Higgs field, leading to localized spacetime metric manipulation and resulting in temporal dilation. This interaction is unstable, causing the CP to decay into tachyons. Conversely, at low energy densities, the UCF field interacts primarily with electroweak forces, specifically influencing hydrogen bond strength without coupling to the Higgs field. Our mathematical framework successfully predicts the linear frequency-dilation scaling observed by Thorne and the temporary nature of the protein stabilization found by Petrova. The theory suggests a critical energy threshold at approximately 500 GHz where the interaction modality shifts from electroweak to Higgs-dominant.

TASK:
Synthesize the findings from the three abstracts into a coherent summary for a government science advisor. The summary should compare and contrast the key findings, explain the theoretical bridge proposed by Dr. Li, and highlight the primary implications of the research.

SUMMARY REPORT:
To: Senior Science Advisor
From: [Your Name]
Subject: Synthesis of Recent Chroniton Particle Research

A review of three recent publications on Chroniton Particles (CPs) reveals a rapidly advancing but complex field. The research highlights two distinct experimental outcomes and a new unifying theory.

Key Experimental Findings:
Dr. Aris Thorne's work at Cambridge focused on high-frequency CPs, discovering that they cause a measurable temporal dilation, making time pass slower within the particle stream. This effect is directly proportional to the CP frequency. A critical side-effect observed was the decay of CPs into a 'tachyon cascade,' suggesting a high-energy, unstable interaction. In contrast, Dr. Lena Petrova's research at MIT explored low-frequency CPs and found a completely different effect: the temporary enhancement of structural stability in organic proteins. This has significant potential for medical applications and, notably, did not produce the dangerous tachyon cascade seen in Thorne's experiments.

The Unifying Theory:
The apparent contradiction between these two results is addressed by Dr. Jian Li's Unified Chroniton Field (UCF) theory. This theoretical framework proposes that CPs behave differently based on their energy levels. Dr. Li suggests that at high energies, CPs interact with the Higgs field to warp spacetime, but at low energies, they interact with the electroweak force to strengthen molecular bonds.

Synthesis and Implications:
Essentially, the research collectively indicates that Chroniton Particles are a highly versatile discovery with a dual nature. The key takeaways are:
1.  **High-Energy vs. Low-Energy:** High-frequency applications manipulate time but come with energy decay and tachyon production, whereas low-frequency applications can safely enhance biomaterials.
2.  **Theoretical Breakthrough:** Dr. Li's UCF theory provides a crucial roadmap for understanding these dual properties and predicts a specific energy threshold (~500 GHz) where the particle's behavior changes dramatically.
3.  **Future Directions:** The primary implication is that future research must focus on precisely controlling CP energy levels to harness their beneficial properties. For medical and biotech applications, research must remain in the low-energy domain to avoid the unstable effects observed by Dr. Thorne. For fundamental physics, exploring the predicted 500 GHz threshold is now the most critical next step.

Further analysis of this threshold could
""",
        "params": {
            "max_tokens": 1500,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n\n"]
        }
    },
    {
        "name": "Complex Test 2: Full-Stack Codebase Analysis and Completion",
        "prompt": """
CONTEXT:
You are tasked with adding a new feature to an existing application. Below is the relevant code from the backend service (Python/Flask), the frontend client (JavaScript/Fetch), and the reverse proxy configuration (Nginx).

File 1: `backend/app.py` (Python Flask Server)
-------------------------------------------------
from flask import Flask, jsonify, request, g
import sqlite3
import time
import redis

app = Flask(__name__)
DATABASE = 'users.db'
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def get_redis():
    r = getattr(g, '_redis', None)
    if r is None:
        r = g._redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    return r

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/api/v1/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    # This endpoint is heavily cached
    r = get_redis()
    cache_key = f"user:{user_id}"
    cached_user = r.get(cache_key)
    if cached_user:
        return jsonify(json.loads(cached_user))

    db = get_db()
    cursor = db.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    
    user_dict = dict(user)
    r.set(cache_key, json.dumps(user_dict), ex=3600) # Cache for 1 hour
    return jsonify(user_dict)

@app.route('/api/v1/users/<int:user_id>/profile', methods=['PUT'])
def update_user_profile(user_id):
    if not request.json or not 'profile_data' in request.json:
        return jsonify({'error': 'Missing profile data'}), 400
    
    profile_data = request.json['profile_data']
    # In a real app, we would validate this data extensively
    
    db = get_db()
    db.execute('UPDATE users SET profile = ? WHERE id = ?', (json.dumps(profile_data), user_id))
    db.commit()
    
    # Important: Invalidate the user cache after an update
    r = get_redis()
    cache_key = f"user:{user_id}"
    r.delete(cache_key)
    
    return jsonify({'status': 'success', 'user_id': user_id})

# ... (other endpoints omitted) ...


File 2: `frontend/api.js` (JavaScript Client)
-------------------------------------------------
const API_BASE_URL = '/api/v1';

async function fetchUserData(userId) {
  try {
    const response = await fetch(`${API_BASE_URL}/users/${userId}`);
    if (!response.ok) {
      if (response.status === 429) {
        console.error("Rate limit exceeded. Please try again later.");
        showRateLimitModal();
      } else {
        console.error(`Error fetching user: ${response.statusText}`);
      }
      return null;
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Network error while fetching user data:", error);
    return null;
  }
}

async function updateUserProfile(userId, profileData) {
  try {
    const response = await fetch(`${API_BASE_URL}/users/${userId}/profile`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ profile_data: profileData }),
    });
    if (!response.ok) {
      console.error(`Error updating profile: ${response.statusText}`);
      return false;
    }
    return true;
  } catch (error) {
    console.error("Network error while updating profile:", error);
    return false;
  }
}


File 3: `nginx/sites-available/default` (Nginx Configuration)
-------------------------------------------------
# Rate limiting zone: 10 megabytes, 1 request per second
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=1r/s;

server {
    listen 80;
    server_name myapp.com;

    location / {
        # Standard frontend proxy
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/v1/ {
        # API backend proxy
        limit_req zone=api_limit burst=5 nodelay; # Allow bursting up to 5 requests
        
        proxy_pass http://localhost:5000; # Flask app
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}


TASK:
The product manager wants a new feature: a "user activity feed" that shows the last 5 actions a user has taken. This is considered sensitive data and should NOT be cached. It also needs its own rate limit to prevent abuse.

Based on the existing codebase, a new function `get_user_activity` is needed. It must:
1.  Connect to the SQLite database.
2.  Query a new `activity_log` table for the last 5 entries for the given `user_id`, ordered by timestamp.
3.  Return a 404 error if the user does not exist in the `users` table.
4.  Implement its own IP-based rate limiting using Redis, separate from the Nginx limit. It should allow 20 requests per minute per IP. The function should return a 429 error if the limit is exceeded.
5.  The function should NOT cache the activity data itself, as it's real-time and sensitive.
6.  The function should be located at the endpoint `/api/v1/users/<int:user_id>/activity`.

Here is the start of the new function:
```python
@app.route('/api/v1/users/<int:user_id>/activity', methods=['GET'])
def get_user_activity(user_id):
    '''Handles the user activity feed request with rate limiting.'''
    # 1. Implement IP-based rate limiting using Redis (20 reqs/min)
    r = get_redis()
    ip_address = request.remote_addr
    rate_limit_key = f"rate_limit:activity:{ip_address}"
    
    current_reqs = r.get(rate_limit_key)
    if current_reqs and int(current_reqs) >= 20:
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    # Use a pipeline for atomic increment and expire
    pipeline = r.pipeline()
    pipeline.incr(rate_limit_key)
    pipeline.expire(rate_limit_key, 60) # Set expiry for 1 minute
    pipeline.execute()

    # 2. Check if the user exists
    db = get_db()
    cursor = db.execute('SELECT id FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if user is None:
        return jsonify({'error': 'User not found'}), 404

    # 3. Query for the last 5 activity log entries
    cursor = db.execute(
        'SELECT action, timestamp, details FROM activity_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5',
        (user_id,)
    )
    activities = [dict(row) for row in cursor.fetchall()]
    
    # 4. Return the data (do not cache)
    
```
""",
        "params": {
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 3: Needle-in-a-Haystack Legal Document Analysis",
        "prompt": """
CONTEXT:
You are a paralegal asked to review the Terms of Service for a new cloud storage provider, "DataSphere Inc." A client is concerned about data ownership and deletion policies. You must find the specific clauses related to their question within the dense document below.

--- START OF DATASPHERE INC. TERMS OF SERVICE ---

**Last Updated: August 1, 2025**

**1. ACCEPTANCE OF TERMS**
This Terms of Service ("TOS") agreement is a legally binding contract between you ("User," "you") and DataSphere Inc. ("DataSphere," "we," "us"). By creating an account, accessing our services, or uploading any data, you acknowledge that you have read, understood, and agree to be bound by this TOS. If you do not agree, you must not use our services.

**2. SERVICE DESCRIPTION**
DataSphere provides cloud storage, file synchronization, and data management services (the "Service"). We reserve the right to modify, suspend, or discontinue the Service at any time with or without notice. We are not liable for any modification, suspension, or discontinuation of the Service.

**3. USER ACCOUNTS**
You must be at least 18 years old to create an account. You are responsible for maintaining the confidentiality of your password and for all activities that occur under your account. You agree to notify us immediately of any unauthorized use of your account.

**4. USER CONDUCT**
You agree not to use the Service to upload, store, or share any data that is unlawful, harmful, threatening, abusive, defamatory, infringing, or otherwise objectionable. You may not use the Service to distribute malware or engage in phishing schemes. Violation of these terms may result in immediate account termination.

**5. INTELLECTUAL PROPERTY**
All intellectual property rights in the Service itself, including the software, branding, and design, are owned by DataSphere Inc. This TOS does not grant you any rights to our intellectual property except for the limited right to use the Service.

**6. USER DATA AND OWNERSHIP**
This section governs your rights to the data you upload to the Service ("User Data").
    6.1. Your Ownership. You retain full ownership of and intellectual property rights to your User Data. We do not claim any ownership over your files.
    6.2. Our License to Your Data. To provide the Service, you grant DataSphere a worldwide, non-exclusive, royalty-free, sublicensable license to use, reproduce, modify, adapt, publish, and distribute your User Data solely for the purpose of operating, providing, and improving the Service. This includes creating thumbnails, transcoding files for different devices, and indexing content for search functionality. This license ends when you delete your User Data or your account, subject to the provisions in Section 7.
    6.3. Data Analytics. Notwithstanding section 6.1, you agree that DataSphere may anonymize and aggregate your User Data with data from other users for statistical analysis, machine learning model training, and service optimization. This anonymized data does not contain personally identifiable information and may be retained by us indefinitely, even after your account is terminated.

**7. DATA RETENTION AND DELETION**
This section details our policies regarding the storage and deletion of your User Data.
    7.1. Active Storage. User Data in an active account is stored across multiple geographically redundant data centers to ensure durability and availability.
    7.2. Deletion Process. When you delete a file from your account, it is moved to a "trash" folder. Files in the trash folder are permanently deleted after 30 days. You can also manually "empty" the trash to expedite this process.
    7.3. Account Termination. Upon termination of your account, either by you or by us, we will initiate the process of deleting your User Data. This process is not instantaneous. For a period of sixty (60) days following account closure, your data may remain on our active servers to allow for account recovery in case of accidental deletion.
    7.4. Backup and Archival Systems. For disaster recovery purposes, our systems create periodic backups. User Data may persist in these offline, encrypted backup archives for a period of up to one (1) year after being deleted from our active systems. These archives are isolated and are not used to serve data to the live Service. Access to this data is strictly controlled and is used only for full system restoration in the event of a catastrophic failure.
    7.5. Legal Holds. We may retain User Data for longer periods than specified above if required by law, regulation, or legal process.

**8. FEES AND PAYMENT**
Service fees are billed on a subscription basis. You agree to pay all applicable fees. Failure to pay may result in suspension or termination of your account.

**9. DISCLAIMER OF WARRANTIES**
THE SERVICE IS PROVIDED "AS IS." WE DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

**10. LIMITATION OF LIABILITY**
IN NO EVENT SHALL DATASPHERE BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES, OR FOR ANY LOSS OF PROFITS OR DATA, ARISING OUT OF YOUR USE OF THE SERVICE.

**11. GOVERNING LAW**
This TOS shall be governed by the laws of the State of California, without regard to its conflict of law principles.

**12. MISCELLANEOUS**
This TOS constitutes the entire agreement between you and DataSphere. If any provision is found to be unenforceable, the remaining provisions will remain in full force and effect.

--- END OF DATASPHERE INC. TERMS OF SERVICE ---


TASK:
The client has read the TOS and has two specific questions. Please provide the answers by continuing the email draft below, based *only* on the text of the TOS provided.

To: client@email.com
From: paralegal@lawfirm.com
Subject: Re: Analysis of DataSphere Terms of Service

Dear Client,

Thank you for your inquiry. I have reviewed the DataSphere Inc. Terms of Service document you provided. Here are the answers to your specific questions:

**1. "After I delete a file, is it gone forever immediately? If not, how long do they keep it?"**

No, your files are not gone forever immediately. The process has several stages according to the TOS:
- First, the deleted file goes into a "trash" folder where it stays for 30 days before automatic permanent deletion (Section 7.2).
- Even after being deleted from the trash and active systems, your data might still exist in their offline, encrypted backup archives for disaster recovery purposes. According to Section 7.4, the data can persist in these archives for up to one year.

**2. "I see they say I 'retain full ownership' of my data, but then they talk about using it for 'machine learning model training'. Can they use my actual documents and photos to train their AI? This is my main concern."**

This is an excellent question, and the distinction is important. According to the TOS:
- You do retain full ownership of your data (Section 6.1).
- However, you grant them a license to use your data to operate and improve the service. Crucially, Section 6.3 states that they have the right to **anonymize and aggregate** your data for purposes including machine learning model training. This means they can use a version of your data that has had all personally identifiable information removed. The TOS explicitly states this anonymized data may be kept and used by them indefinitely, even after you close your account.

In summary, they will not use your raw, identifiable documents and photos for AI training, but they
""",
        "params": {
            "max_tokens": 800,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n\n"]
        }
    },
    {
        "name": "Complex Test 4: Long-Form Narrative Reasoning and Deduction",
        "prompt": """
CONTEXT:
You are a detective summarizing your final deductions in a complex murder case. The victim is Lord Harrington, a wealthy industrialist. The murder occurred in his locked study. Below is the full case file, including suspect interviews and forensic reports.

--- START OF CASE FILE: THE HARRINGTON MURDER ---

**Victim:** Lord Harrington, age 68.
**Cause of Death:** Poisoning (fast-acting neurotoxin, Tetrodotoxin).
**Time of Death:** Estimated between 9:00 PM and 10:00 PM.
**Scene:** The victim's study. The door was locked from the inside. The only window was sealed shut with old paint. A half-empty glass of whiskey sat on his desk next to an open bottle.

**Forensic Report:**
-   The whiskey in the glass contained a lethal dose of Tetrodotoxin. The bottle of whiskey was clean.
-   No foreign fingerprints were found on the glass, only the victim's.
-   A tiny, almost invisible puncture mark was found on the cork of the whiskey bottle.
-   The victim's clothes had a faint smell of almonds, which is odd as Tetrodotoxin is odorless. The poison in the glass was pure, with no almond scent.
-   A single, out-of-place peacock feather was found under the victim's desk.

**Suspects & Alibis:**

**1. Lady Beatrice Harrington (Wife, age 45)**
-   **Motive:** Stood to inherit the entire Harrington fortune. Known to be in significant personal debt from gambling. Was having an affair.
-   **Alibi:** Claims she was in the estate's greenhouse from 8:30 PM until 10:30 PM, tending to her exotic plants.
-   **Interview Notes:** Seemed distraught but calm. Mentioned that Lord Harrington had recently changed his will, but didn't know the details. She wears a peacock feather fascinator to social events. When asked about it, she stated it was locked in her jewelry box all night.

**2. Edward Finch (Business Rival, age 55)**
-   **Motive:** Lord Harrington was about to expose Finch for corporate espionage, which would have ruined him.
-   **Alibi:** Was at a public gala across town. His presence is confirmed by dozens of witnesses from 8:00 PM until midnight. He left the gala only once, for about 10 minutes around 9:15 PM, to take a phone call in the lobby.
-   **Interview Notes:** Arrogant and dismissive. Admitted to hating Harrington but laughed at the accusation. His tie pin is a small, silver feather. He has a chemistry background from his university days.

**3. Charles "Charlie" Croft (Butler, age 62)**
-   **Motive:** Recently fired by Lord Harrington after 40 years of service for a minor mistake. He was due to leave the estate the next day.
-   **Alibi:** Was in the servants' quarters, packing his belongings. His roommate, a maid, confirms he was there from 8:00 PM onwards, but she admits she fell asleep around 9:00 PM and didn't wake until 11:00 PM.
-   **Interview Notes:** Visibly bitter and resentful. Spoke of his loyalty being repaid with cruelty. He is the only one, other than Harrington, with a key to the study, though Harrington's was found on his person and the door was bolted from the inside. Charlie mentioned that the study's lock was old and could sometimes be "jiggled" open without a key if you knew the trick. He would have served Lord Harrington his nightly whiskey.

**4. Dr. Alistair Reed (Personal Physician, age 38)**
-   **Motive:** Lord Harrington had discovered Dr. Reed was falsifying medical research and threatened to expose him. Dr. Reed has access to a wide variety of toxins and chemicals.
-   **Alibi:** Was at his clinic, seeing late-night patients. His receptionist confirms he was there until 11:00 PM.
-   **Interview Notes:** Extremely nervous and fidgety. He has a pet peacock at his countryside home. He mentioned that he had visited Lord Harrington earlier in the day, around 4:00 PM, for a routine check-up and had brought him a "calming tea" laced with a harmless cyanide compound (which can smell of almonds) to help him sleep. He insisted it was a micro-dose and couldn't have killed him.

--- END OF CASE FILE ---

TASK:
Complete the detective's final, conclusive monologue, explaining who the killer is, how they did it, and how the clues fit together.

**Detective's Monologue:**

"This case appeared to be a classic locked-room mystery, but the key isn't the lock; it's the poison. We have four suspects, each with a motive, but their alibis seem solid. However, the clues, when pieced together, point to only one person.

The murder was a two-part chemical deception. The faint smell of almonds on the victim was a red herring, planted to mislead us. Dr. Reed admitted to giving the victim a tea with a harmless cyanide compound earlier in the day. This explains the smell but has nothing to do with the actual murder. It was a clever, but ultimately telling, attempt to frame the doctor.

The real poison was the odorless Tetrodotoxin, administered through the whiskey. But how was it introduced into a glass that only had the victim's fingerprints, from a bottle that was clean? The tiny puncture mark on the cork is the answer. The poison was injected through the cork into the bottle *before* it was even brought to the study. The killer knew Lord Harrington's routine—that he drank a glass of whiskey every night.

So, who had the means, motive, and opportunity to orchestrate this?

Lady Beatrice had motive, and the peacock feather seems to point to her, but it's too obvious. It was planted. Edward Finch had a strong motive but an even stronger alibi. Being at a public gala makes it nearly impossible. Dr. Reed had the means but his alibi is also firm, and his admission about the cyanide tea seems more like panicked honesty than deception.

This leaves the butler, Charlie Croft. His motive is one of deep-seated revenge. His alibi is the weakest, relying on a sleeping roommate. He had access to the whiskey bottle long before it reached the study. He could have injected the poison hours earlier. He knew the study's lock could be bypassed, allowing him to plant the peacock feather under the desk to frame Lady Beatrice after the murder.

But the final, damning piece of evidence is the combination of the injected cork and the almond scent. The killer needed to know about the doctor's visit and the cyanide tea to use it as a misdirection. Who would know about the doctor's 4 PM visit and the special 'calming tea' he brought? Not the wife in her greenhouse, nor the rival across town. It was the butler, Charlie Croft, who would have been present, serving the tea and overhearing the conversation. He used that knowledge to create a perfect storm of confusion. He poisoned the whiskey, planted the feather to implicate the wife, and relied on the almond scent from the tea to pointlessly implicate the doctor. It was a crime of a man who felt betrayed, using his intimate knowledge of the household to try and create the perfect unsolvable murder. He knew the habits of everyone involved and used that knowledge to
""",
        "params": {
            "max_tokens": 1200,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n---"]
        }
    },
    {
        "name": "Complex Test 5: Complex Technical Manual Troubleshooting",
        "prompt": """
CONTEXT:
You are a senior network engineer for "QuantumLink Inc." A junior technician has escalated a support ticket for a client's "Continuum Router 9000" (CR-9000). You must use the provided excerpts from the official CR-9000 Technical Manual to formulate a step-by-step solution.

--- START OF CR-9000 TECHNICAL MANUAL EXCERPTS ---

**Section 2: System Architecture**
...The CR-9000 operates with a dual-firmware partition system: Partition A (Active) and Partition B (Recovery). The active firmware is loaded from Partition A on boot. If the bootloader detects a checksum failure on Partition A three consecutive times, it will automatically attempt to boot from Partition B. The `sys-led` indicator will flash amber during a recovery boot. Standard boot is a solid green light... The device's configuration files are stored in a separate NVRAM module and are not tied to a specific firmware partition. However, firmware version 3.0 and newer uses a new `config.v2` format. Older firmware (v2.x) cannot read this format. If booting on older firmware with a `config.v2` file, the router will fail to load its configuration and will enter a "default state" with factory settings (IP: 192.168.1.1).

**Section 4: Command Line Interface (CLI) Guide**
...Access the CLI via SSH on port 22 or the serial console port.
-   `show version`: Displays the firmware version on both Partition A and Partition B.
-   `show status`: Displays uptime, CPU load, memory usage, and `sys-led` status.
-   `reboot partition <A|B>`: Forces the router to reboot using the specified partition. Use with caution.
-   `upload firmware <url>`: Downloads firmware from a URL and installs it to the INACTIVE partition. This command automatically runs a checksum verification post-download.
-   `set active-partition <A|B>`: Manually switches the active boot partition for the next reboot.
-   `factory-reset`: Wipes the configuration in NVRAM and reboots the device. This does not affect the installed firmware partitions.
-   `show logs -f`: Tails the system logs in real-time. Look for `CONFIG_LOAD_FAIL` errors.

**Section 7: Troubleshooting Common Issues**
-   **Issue: No Connectivity after Firmware Update.**
    -   **Cause A:** The update may have failed checksum verification. Check logs.
    -   **Cause B:** IP address may have changed. Connect via serial console to verify network settings.
    -   **Cause C:** Configuration incompatibility. This is common when downgrading firmware. If a new configuration format was used, the older firmware may not be able to read it, causing the router to revert to a factory default IP (192.168.1.1).
-   **Issue: `sys-led` is Flashing Amber.**
    -   **Cause:** The device has failed to boot from the active partition multiple times and is now running on the recovery partition. This firmware is often an older, stable version. Performance may be degraded. It is recommended to re-flash the active partition.

**Appendix B: Firmware Release Notes**
-   **Version 3.2 (Latest):** `cr9k-firmware-v3.2.bin` - Includes security patches and performance improvements. Uses `config.v2` format.
-   **Version 3.1:** `cr9k-firmware-v3.1.bin` - First release with `config.v2` support.
-   **Version 2.8 (Stable Recovery):** `cr9k-firmware-v2.8.bin` - Cannot read `config.v2`. This is the default firmware on Partition B.

--- END OF CR-9000 TECHNICAL MANUAL EXCERPTS ---

**SUPPORT TICKET DETAILS:**
- **Client:** MegaCorp Inc.
- **Device:** Continuum Router 9000
- **Technician Notes:** "Client attempted to update the firmware to the latest version (v3.2) last night. This morning, they reported a total network outage. I can't ping the router at its static IP (10.10.1.1). When I had them check the device, they said the main system light is flashing amber. I'm not sure what to do next."

TASK:
Based on all the provided information (manual excerpts and ticket details), write a clear, step-by-step recovery plan for the junior technician to follow. The plan should explain the likely cause of the problem and provide the exact CLI commands needed to fix it.

**Recovery Plan for Ticket #734 - MegaCorp Inc.**

**Prepared by:** Senior Network Engineer
**For:** Junior Technician

**1. Problem Analysis (What Happened):**
The evidence strongly suggests a cascading failure caused by the recent firmware update. Here's the likely sequence of events:
- The client uploaded the new v3.2 firmware. This update likely failed its checksum verification or was corrupted, rendering Partition A (the active partition) unbootable.
- After failing to boot from Partition A three times, the router automatically switched to its recovery partition, Partition B, as per its design (Section 2). This is confirmed by the client's report of a flashing amber `sys-led` (Section 7).
- Partition B contains the older, stable firmware v2.8 (Appendix B).
- The router's configuration, which includes the static IP of 10.10.1.1, was saved in the new `config.v2` format by the v3.2 firmware before it failed.
- The now-active v2.8 firmware cannot read the `config.v2` format (Section 2, Appendix B). This caused a `CONFIG_LOAD_FAIL` error, forcing the router to revert to its factory default settings, including a default IP address of 192.168.1.1 (Section 7, Cause C).

This is why you cannot ping it at 10.10.1.1. The router is online but on a different IP and without its proper configuration.

**2. Step-by-Step Recovery Procedure:**
Have the technician follow these steps precisely. They will need to connect a laptop directly to the router's serial console port.

**Step 1: Connect and Verify the Situation**
   - Instruct the technician to establish a serial console connection to the CR-9000.
   - Once connected, have them run the `show version` command. This will confirm our hypothesis. The output should show Partition B as active and running v2.8, while Partition A has v3.2.
   - Next, run `show status`. This will confirm the `sys-led` status is "Flashing Amber".

**Step 2: Re-flash the Corrupted Firmware on Partition A**
   - The goal is to install a clean version of the latest firmware onto the inactive partition (A).
   - Provide the technician with the download URL for the firmware: `http://firmware.quantumlink.com/cr9k-firmware-v3.2.bin`
   - Have them execute the following command:
     ```
     upload firmware [http://firmware.quantumlink.com/cr9k-firmware-v3.2.bin](http://firmware.quantumlink.com/cr9k-firmware-v3.2.bin)
     ```
   - The router will download and install the firmware onto Partition A, which is currently inactive. It will automatically verify the checksum. This process may take several minutes.

**Step 3: Switch the Active Partition and Reboot**
   - Once the upload is complete and verified, we need to tell the router to boot from the newly flashed Partition A on its next startup.
   - Execute the command:
     ```
     set active-partition A
     ```
   - Now, reboot the router to apply the changes:
     ```
     reboot
     ```

**Step 4: Final Verification**
   - After the router reboots, the `sys-led` should turn solid green.
   - The router will now be running the clean v3.2 firmware from Partition A. This version can correctly read the `config.v2` file, so it will load its proper configuration and be available at its static IP of 10.10.1.1.
   - Have the technician disconnect from the serial console and attempt to ping the router at 10.10.1.1. They should also be able to access it via SSH.
   - Finally, have them run `show version` one last time via SSH to confirm that Partition A is now the active partition.

This procedure should fully resolve the outage. Please monitor the ticket and
""",
        "params": {
            "max_tokens": 1500,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n---"]
        }
    },
    # ======================================================================================
    # NEW 10K+ TOKEN TEST CASES
    # ======================================================================================
    {
        "name": "Complex Test 6: Grand Strategy Wargame Scenario and Analysis",
        "prompt": """
CONTEXT:
You are the Supreme Strategic Advisor for the fictional nation of Eldoria. A major geopolitical crisis is unfolding. You must analyze the provided intelligence dossier and draft a comprehensive grand strategy document for the Eldorian High Council.

--- INTELLIGENCE DOSSIER: THE AETHELIAN CRISIS ---

**1. Belligerents:**
* **The Republic of Eldoria:** A technologically advanced, maritime trading nation with a powerful navy but a relatively small, professional army. Relies heavily on imported rare earth minerals. Democratic government, values economic stability and international law.
* **The Valerium Empire:** An expansionist, autocratic continental power with a massive, conscript-based army and significant armored divisions. Strong industrial base but technologically a generation behind Eldoria. Aims to secure resource-rich territories and establish regional hegemony.
* **The Kingdom of Caelus:** A mountainous, isolationist nation with a highly trained, elite air force and formidable anti-air defenses. Controls the primary mountain passes between Eldoria and Valerium. Officially neutral but has a historical non-aggression pact with Eldoria.
* **The Free State of Koth:** A small, resource-rich nation located on a strategic peninsula. Possesses the world's largest deposits of "Aetherium," the rare earth mineral essential for Eldoria's advanced technology. Has a weak military and is currently under intense political pressure from the Valerium Empire.

**2. Geographical & Economic Factors:**
* The "Strait of Whispers" is the only viable sea lane for Eldorian trade. It is narrow and could be easily blockaded.
* The "Dragon's Tooth Mountains" separate Eldoria and Valerium. Only two major passes exist, both controlled by Caelus. A land invasion is impossible without Caelus's cooperation.
* Eldoria's economy will collapse within 6 months if Aetherium imports from Koth are severed. Valerium's economy is slower but more resilient and can sustain a prolonged conflict.
* Valerium has recently constructed a "Sky Shield" anti-air network along its border, making a direct air assault by Caelus's forces a high-risk endeavor.

**3. Military Dispositions:**
* **Eldoria:**
    * Navy: 3 Carrier Strike Groups (CSGs), 12 advanced stealth destroyers, 20 frigates, 8 nuclear attack submarines. Deployed primarily to protect the Strait of Whispers and sea lanes to Koth.
    * Army: 2 mechanized infantry divisions, 1 special forces brigade. Positioned defensively near the Caelus border.
    * Air Force: Small but modern, focused on maritime patrol and fleet air defense.
* **Valerium:**
    * Army: 15 infantry divisions, 5 armored divisions, 3 artillery divisions. Massed along the borders of Koth and Caelus.
    * Navy: Primarily a coastal defense force, no match for Eldoria in open water.
    * Air Force: Large but outdated. Would suffer heavy losses against Caelus or Eldorian naval fighters.
* **Caelus:**
    * Air Force: 25 squadrons of elite "Gryphon" interceptors, extensive mountain-based SAM sites.
    * Army: 4 mountain infantry divisions, experts in defensive warfare.

**4. The Spark:**
The Valerium Empire has issued an ultimatum to the Free State of Koth: accept "protective annexation" within 30 days or face a full-scale invasion. Eldoria has a mutual defense treaty with Koth. The international community has condemned Valerium but has not committed to military intervention.

TASK:
Draft a comprehensive, multi-phase grand strategy document for the Eldorian High Council. The document must be exceptionally detailed and long-form. It should include:

1.  **Threat Assessment:** A detailed analysis of the strategic threat posed by the Valerium Empire, considering their strengths, weaknesses, and most likely course of action.
2.  **Strategic Objectives:** A clear definition of Eldoria's primary, secondary, and tertiary strategic goals in this conflict. (e.g., preservation of Koth's sovereignty, securing Aetherium supply, degrading Valerium's military, etc.).
3.  **Proposed Grand Strategies (Present 3 Options):**
    * **Option A: "The Maritime Shield."** A defensive strategy focused on naval blockade and economic strangulation of Valerium.
    * **Option B: "The Caelian Gambit."** A diplomatic and military strategy focused on convincing Caelus to intervene, opening a second front.
    * **Option C: "The Kothian Dagger."** A high-risk, pre-emptive strategy involving rapid deployment of Eldorian forces to Koth to repel the initial invasion and force a stalemate.
4.  **In-Depth Analysis of Strategies:** For each of the three options, provide a thorough analysis covering:
    * Required military and diplomatic actions.
    * Potential costs in terms of military assets, economic impact, and human life.
    * Likely Valerian counter-moves.
    * The role of Caelus in the strategy.
    * The probability of success and the definition of "victory" under this strategy.
5.  **Recommended Strategy & Phased Operational Plan:**
    * Select one of the three strategies as your primary recommendation.
    * Provide a detailed, multi-phase operational plan for your recommended strategy. This plan should cover the initial 90 days of the conflict and include specific actions for the Navy, Army, Air Force, and Diplomatic Corps. It should be broken down into:
        * **Phase 1: Pre-Conflict (Days 0-30):** Deterrence, deployment, and diplomacy.
        * **Phase 2: Initial Hostilities (Days 31-60):** The opening moves of the war.
        * **Phase 3: Escalation & Attrition (Days 61-90):** Responding to Valerian counter-attacks and setting conditions for conflict termination.
6.  **Long-Term Implications & Post-Conflict Vision:** Discuss the potential long-term geopolitical consequences of your recommended strategy and outline a vision for the post-conflict regional order that favors Eldorian interests.

This document must be thorough, well-reasoned, and demonstrate a deep understanding of the complex interplay between military, economic, and diplomatic factors.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF REPORT ---"]
        }
    },
    {
        "name": "Complex Test 7: Full-Length Scientific Research Paper from Fictional Data",
        "prompt": """
CONTEXT:
You are a senior research scientist at a bio-engineering firm. You have just concluded a series of experiments on a novel enzyme, "Cryo-Catalase 7" (CC-7), which appears to grant organisms resistance to freezing temperatures. You must now write a full-length scientific research paper for submission to the journal "Cellular Biology & Cryogenics."

--- EXPERIMENTAL DATA & LAB NOTES ---

**Project Title:** Elucidating the Mechanism of Cryo-Catalase 7 (CC-7) in E. coli

**Background:**
* CC-7 was isolated from *Psychrobacter arcticus*, a bacterium found in arctic permafrost.
* Hypothesis: CC-7 functions as an "ice-structuring protein," preventing the formation of large, cell-damaging ice crystals within the cytoplasm.

**Experiment 1: CC-7 Expression in E. coli**
* Method: The gene for CC-7 was inserted into a plasmid and expressed in a standard E. coli (BL21 strain). A control group was prepared with an empty plasmid.
* Results: Western blot analysis confirmed high levels of CC-7 expression in the experimental group. No expression was seen in the control group. Both groups showed identical growth rates at 37°C.

**Experiment 2: Cryo-Survival Assay**
* Method: Cultures of both experimental (E. coli + CC-7) and control E. coli were subjected to a rapid freeze-thaw cycle (-80°C for 1 hour, then thawed to 37°C). Cell viability was measured by counting colony-forming units (CFUs) before and after the cycle.
* Data Table:
| Group             | Pre-Freeze CFU/mL | Post-Thaw CFU/mL | Survival Rate |
|-------------------|-------------------|------------------|---------------|
| Control (n=3)     | 8.2 x 10^8        | 1.5 x 10^4       | 0.0018%       |
| E. coli + CC-7 (n=3)| 8.5 x 10^8        | 7.9 x 10^8       | 92.9%         |
* Lab Note: "The difference is staggering. The CC-7 plates are almost identical to the pre-freeze plates. The control is basically a wasteland."

**Experiment 3: Microscopic Analysis of Ice Crystal Formation**
* Method: Using a cryo-electron microscope, we observed the cytoplasm of both groups during a controlled freezing process.
* Observations:
    * **Control E. coli:** As temperatures dropped below 0°C, large, needle-like ice crystals formed rapidly, visibly rupturing the cell membranes.
    * **E. coli + CC-7:** Ice crystal formation was significantly different. Instead of large needles, a network of tiny, hexagonal, micro-crystals formed. These micro-crystals did not appear to damage the cell membrane or internal organelles. The cytoplasm took on a vitrified, glass-like appearance.
* Lab Note: "It's not stopping the freezing, it's *controlling* it. The CC-7 seems to be acting as a nucleation point for millions of tiny, harmless crystals."

**Experiment 4: In Vitro Ice Recrystallization Inhibition (IRI) Assay**
* Method: Purified CC-7 protein was added to a sucrose solution on a microscope slide and frozen. The sample was then held at -6°C (a temperature that promotes recrystallization) for 24 hours. A control sample with Bovine Serum Albumin (BSA), a non-cryoprotective protein, was also tested.
* Results:
    * **BSA Control:** Over 24 hours, small ice crystals grew into large, multi-faceted crystals, demonstrating significant recrystallization.
    * **CC-7 Sample:** The initial small ice crystals showed almost no change in size or shape over the 24-hour period.
* Lab Note: "This confirms the microscopic observations. CC-7 is a potent inhibitor of ice recrystallization. This is likely the core of its cryoprotective mechanism."

**Experiment 5: Structural Analysis of CC-7**
* Method: X-ray crystallography was used to determine the 3D structure of the CC-7 protein.
* Findings: The protein has a unique, beta-solenoid structure. One face of the protein is remarkably flat and features a highly repetitive, hydrophilic pattern of threonine and serine residues.
* Hypothesis Update: "This flat, repetitive surface is almost certainly the ice-binding face. The spacing of the hydrophilic residues likely matches the lattice structure of ice, allowing it to bind to nascent ice crystals and prevent their further growth."

TASK:
Write a complete, formal scientific research paper based on the provided data and notes. The paper must be structured correctly and written in a formal, academic tone. It must be exceptionally detailed and comprehensive.

**Required Sections:**
1.  **Title:** A concise, descriptive title for the paper.
2.  **Authors and Affiliations:** (Use fictional names).
3.  **Abstract:** A brief summary (approx. 250 words) of the study's background, methods, key results, and conclusion.
4.  **Introduction:** A detailed background on the challenges of cellular cryopreservation, the role of ice-structuring proteins in nature, and the specific objectives of this study. This section should be long and cite fictional prior research.
5.  **Materials and Methods:** A thorough description of all experimental procedures. Provide enough detail that another scientist could replicate the experiments. This includes plasmid construction, cell culture conditions, the cryo-survival assay protocol, microscopy parameters, the IRI assay, and the crystallography methods.
6.  **Results:** A comprehensive presentation of the findings from all five experiments. Describe the data from the tables and the observations from the microscopic analyses in detail. Refer to figures (e.g., "Figure 1A," "Figure 2B") as if they were present in the paper.
7.  **Discussion:** A long and in-depth analysis of the results. This is the most important section.
    * Interpret the findings: Explain what the results mean in the context of the initial hypothesis.
    * Synthesize the data: Connect the results from the different experiments to build a cohesive model of how CC-7 works (e.g., "The high survival rate seen in Experiment 2 is explained by the controlled micro-crystal formation observed in Experiment 3...").
    * Propose a detailed molecular mechanism: Based on the structural data from Experiment 5, explain exactly how the flat, hydrophilic face of CC-7 likely binds to ice crystals and inhibits their growth.
    * Discuss the implications of the findings for fields like cryobiology, medicine (organ preservation), and biotechnology.
    * Acknowledge limitations of the study and suggest future research directions.
8.  **Conclusion:** A brief, powerful summary of the study's main takeaway.
9.  **References:** (List at least 10 fictional, but plausibly titled, references).
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF MANUSCRIPT ---"]
        }
    },
    {
        "name": "Complex Test 8: Corporate Merger Integration Playbook",
        "prompt": """
CONTEXT:
You are a senior consultant at a top-tier management consulting firm. Your team has been hired to create a comprehensive post-merger integration playbook for the acquisition of "Momentum Motors" (a legacy automaker) by "Innovate EV" (a high-tech electric vehicle company).

--- PRE-MERGER DOSSIER ---

**Acquirer: Innovate EV Inc.**
* **Founded:** 2015
* **Headquarters:** Silicon Valley, CA
* **Culture:** Fast-paced, agile, "move fast and break things" mentality. Flat organizational structure, heavy reliance on Slack and collaborative software. Engineering-driven. High tolerance for risk.
* **Technology Stack:** Fully cloud-native (AWS). Custom-built ERP system ("Helios"). Data-centric, extensive use of AI and machine learning in manufacturing and R&D.
* **Products:** High-end electric sedans and SUVs with cutting-edge battery technology and autonomous driving features. Direct-to-consumer sales model.
* **Financials:** Market Cap $150B. High growth, but not yet consistently profitable.
* **Workforce:** 15,000 employees, primarily software engineers, data scientists, and robotics experts. Non-unionized.

**Target: Momentum Motors Corp.**
* **Founded:** 1925
* **Headquarters:** Detroit, MI
* **Culture:** Traditional, hierarchical, process-oriented. Formal communication channels. Manufacturing-excellence driven. Low tolerance for risk.
* **Technology Stack:** On-premise data centers. Legacy mainframe systems for core operations. SAP for ERP. Many siloed, departmental software solutions.
* **Products:** Full range of internal combustion engine (ICE) cars, trucks, and SUVs. Strong brand recognition and vast dealership network.
* **Financials:** Market Cap $40B. Low growth, but consistently profitable with strong cash flow.
* **Workforce:** 85,000 employees, primarily mechanical engineers, manufacturing line workers, and sales/marketing professionals. Heavily unionized (United Auto Workers - UAW).

**The Deal:**
* Innovate EV is acquiring Momentum Motors for $60B in a cash and stock deal.
* **Strategic Rationale (for Innovate EV):**
    1.  Gain access to Momentum's massive manufacturing capacity to scale EV production.
    2.  Leverage Momentum's established global dealership and supply chain network.
    3.  Acquire valuable brand equity and a loyal customer base.
* **Key Challenges:**
    1.  Massive cultural clash between the two companies.
    2.  Integration of vastly different technology stacks (Cloud vs. On-premise, modern vs. legacy).
    3.  Managing the unionized workforce of Momentum Motors.
    4.  Product portfolio rationalization (phasing out ICE vehicles).

TASK:
Create a comprehensive, detailed, and long-form Post-Merger Integration (PMI) Playbook. This document will be the master guide for the Integration Management Office (IMO) for the first 18 months post-acquisition.

**Required Sections of the Playbook:**

1.  **Executive Summary:** A high-level overview of the integration strategy, key objectives, major challenges, and the expected timeline for achieving synergies.
2.  **Integration Guiding Principles:** A set of core principles that will guide all integration decisions (e.g., "Best of Both," "Innovate EV Way," "Culture of Collaboration," "Customer-First," etc.). Justify your choice of principles.
3.  **Integration Governance & IMO Structure:**
    * Propose a detailed organizational chart for the Integration Management Office (IMO).
    * Define the roles and responsibilities of the IMO, the Steering Committee (with executives from both companies), and the various functional integration teams (e.g., IT, HR, Manufacturing).
    * Outline the meeting cadence, reporting structure, and decision-making processes.
4.  **Culture & Change Management Plan:**
    * Identify the top 5 cultural risks.
    * Develop a detailed, phased plan to bridge the cultural gap. This should include specific initiatives like joint leadership off-sites, a new combined company values statement, employee town halls, and a "cultural ambassador" program.
    * Outline a comprehensive communication plan for the first 90 days.
5.  **Technology & IT Integration Roadmap:**
    * Provide a detailed strategy for integrating the two technology stacks.
    * Address the core systems: What is the plan for the ERP systems (Helios vs. SAP)? What is the strategy for moving Momentum's operations to the cloud?
    * Create a phased timeline (e.g., Day 1 readiness, 6-month goals, 18-month goals) for key IT milestones like network integration, email system unification, and data migration.
6.  **Manufacturing & Supply Chain Consolidation:**
    * Outline a plan to re-tool Momentum's factories for EV production.
    * Describe how to merge Innovate EV's agile, data-driven manufacturing processes with Momentum's traditional, scaled operations.
    * Detail a strategy for consolidating suppliers and leveraging the combined entity's purchasing power.
7.  **Human Resources & Labor Relations Strategy:**
    * Propose a new, unified organizational structure for the combined company.
    * Outline a process for talent assessment and retention of key employees from both companies.
    * Develop a detailed strategy for working with the UAW. How will Innovate EV approach negotiations for re-training programs, factory transitions, and potential redundancies?
8.  **Sales, Marketing, & Brand Strategy:**
    * What is the go-forward brand strategy? Will the "Momentum Motors" brand continue to exist?
    * How will the direct-to-consumer model of Innovate EV be integrated with Momentum's dealership network? Propose a hybrid model.
    * Outline a plan for rationalizing the combined product portfolio.
9.  **Synergy Realization & Tracking:**
    * Identify the top 5 cost synergies and the top 3 revenue synergies.
    * Create a detailed financial model (in table format) showing the timeline and expected value of these synergies over the next 3 years.
    * Describe the metrics and KPIs that the IMO will use to track the success of the integration.
10. **Risk Mitigation Plan:**
    * Create a risk register (in table format) identifying at least 10 major integration risks.
    * For each risk, assess its likelihood and impact, and propose a detailed mitigation strategy.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF PLAYBOOK ---"]
        }
    },
    {
        "name": "Complex Test 9: Novel Chapter Generation with Deep Character Arcs",
        "prompt": """
CONTEXT:
You are an author writing the next chapter of a dark fantasy novel, "The Ember Blade." You must write Chapter 12, ensuring it is consistent with the provided character backstories and the events of the preceding chapter. The chapter needs to be exceptionally long and detailed, focusing on character development and advancing multiple plotlines.

--- WORLD LORE & CHARACTER DOSSIER ---

* **The World:** A kingdom slowly succumbing to "The Blight," a magical corruption that twists nature and drives people mad. The Blight originates from the fallen Shadowstone, a celestial object that crashed centuries ago.
* **Kaelen ("The Marked"):** The protagonist. A former royal knight, framed for the murder of the king and now a fugitive. He bears a magical mark that allows him to sense the Blight, but it also slowly corrupts him, causing him pain and giving him terrifying visions. He is cynical and guilt-ridden, but deeply honorable. He seeks the Sunstone, a legendary artifact said to be the only thing that can counter the Shadowstone's influence.
* **Seraphina ("The Scholar"):** A brilliant but disgraced scholar from the Royal Archives. She was exiled for studying forbidden texts related to the Blight. She is pragmatic, resourceful, and often clashes with Kaelen's idealism. Her primary motivation is to understand the Blight, believing knowledge is the only path to victory. She carries a collection of ancient maps and texts.
* **Garrick ("The Veteran"):** An old, grizzled former soldier and Kaelen's mentor. He lost his family to the Blight years ago and is now a grim, fatalistic warrior. He is fiercely loyal to Kaelen but fears the mark is changing him. He is a master tactician and survivalist.

--- SUMMARY OF CHAPTER 11: "THE WHISPERING FEN" ---

Kaelen, Seraphina, and Garrick have just escaped a harrowing encounter with a "Blight-beast" in the Whispering Fen. During the fight, Kaelen had to draw upon the power of his mark to defeat the creature, causing him immense pain and a terrifying vision of a "Throne of Rot." The vision left him weak and shaken. Seraphina managed to collect a sample of the beast's corrupted ichor for study. Garrick is growing more concerned about Kaelen's reliance on the mark. They have taken refuge in a hidden, ancient ruin deep within the fen, which Seraphina identified from her maps as a potential "Sun-temple," a place of old power that might be resistant to the Blight. The chapter ended as they barred the stone door to the ruin, finally safe for the moment.

TASK:
Write Chapter 12 of "The Ember Blade." The chapter must be a minimum of 4,000 words.

**Key Objectives for Chapter 12:**

1.  **Explore the Ruin:** The characters must explore the Sun-temple. Describe its architecture, its history (as deciphered by Seraphina), and the atmosphere within. Is it truly safe from the Blight? What secrets does it hold?
2.  **Character Conflict & Development (The Core of the Chapter):**
    * **Kaelen's Internal Struggle:** Kaelen is at a low point. He is physically and mentally exhausted from using the mark. The chapter must feature a deep, introspective dive into his psyche. He should have a significant conversation with Garrick about the vision of the "Throne of Rot" and his fear of what he is becoming. He is torn between his duty and the corrupting influence of his power.
    * **Seraphina's Discovery:** While Kaelen and Garrick deal with the fallout of the battle, Seraphina must analyze the Blight-beast's ichor using her scholar's kit. She should make a groundbreaking, and perhaps disturbing, discovery about the nature of the Blight. Perhaps it's not just a mindless corruption, but something more intelligent or purposeful.
    * **The Central Argument:** The chapter must build to a major philosophical and strategic argument between the three characters.
        * Garrick, seeing Kaelen's suffering, will advocate for a more cautious approach, perhaps even abandoning the quest. He represents survival and protecting what's left.
        * Seraphina, armed with her new discovery, will argue for a more aggressive, knowledge-seeking path. She believes they must understand the enemy to defeat it, even if it means taking risks.
        * Kaelen is caught in the middle. He must make a difficult choice that will define his leadership and the direction of their quest.
3.  **Advance the Plot:** The chapter must end with a clear decision and a new direction. This could be the discovery of a clue within the temple, a new interpretation of Seraphina's maps, or the resolution of their argument leading to a new destination. The ending should feel earned and set a clear path for Chapter 13.
4.  **Maintain Tone and Style:** The writing should be immersive, atmospheric, and dark, consistent with the genre. Use rich descriptions and internal monologue to convey the characters' states of mind.

**Chapter 12 Title:** "The Sun-Temple's Silence"
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF CHAPTER ---"]
        }
    },
    {
        "name": "Complex Test 10: Philosophical Treatise Synthesis and Critique",
        "prompt": """
CONTEXT:
You are a doctoral candidate in philosophy preparing a chapter for your dissertation on the concept of "Power." You must analyze, synthesize, and critique the provided excerpts from three seminal thinkers: Niccolò Machiavelli, Friedrich Nietzsche, and Michel Foucault.

--- EXCERPTED PHILOSOPHICAL TEXTS ---

**Excerpt 1: Niccolò Machiavelli - *The Prince* (c. 1513)**
"...it is necessary for a prince to learn how not to be good, and to use this knowledge and not use it, according to the necessity of the case... For a man who wishes to make a profession of goodness in everything must necessarily come to ruin among so many who are not good. Hence it is necessary for a prince who wishes to maintain himself to learn how to be able to be not good, and to use it and not use it according to necessity...

...a prince must not mind incurring the charge of cruelty for the purpose of keeping his subjects united and faithful; for, with a very few examples, he will be more merciful than those who, from excess of tenderness, allow disorders to arise, from whence spring murders or robberies...

...And here comes in the question whether it is better to be loved rather than feared, or feared rather than loved. It may be answered that one should wish to be both, but, because it is difficult to unite them in one person, it is much safer to be feared than loved, when, of the two, one must be dispensed with. Because this is to be asserted in general of men, that they are ungrateful, fickle, false, cowardly, covetous, and as long as you succeed they are yours entirely... but when need is at hand they turn against you... Love is preserved by the link of obligation which, owing to the baseness of men, is broken at every opportunity for their advantage; but fear preserves you by a dread of punishment which never fails."

**Excerpt 2: Friedrich Nietzsche - *On the Genealogy of Morality* (1887) & *The Will to Power* (notes)**
"What is good? — All that heightens the feeling of power, the will to power, power itself in man. What is bad? — All that proceeds from weakness... The weak and ill-constituted shall perish: first principle of our philanthropy. And one shall help them to do so...

...This world is the will to power—and nothing besides! And you yourselves are also this will to power—and nothing besides!... My idea is that every specific body strives to become master over all space and to extend its force (—its will to power:) and to thrust back all that resists its extension. But it continually encounters similar efforts on the part of other bodies and ends by coming to an arrangement ('union') with those of them that are sufficiently related to it: thus they then conspire together for power. And the process goes on...

...'Master morality' and 'slave morality'... The master's morality is affirmative, it says 'yes' to life; it creates values. The noble type of man experiences itself as determining values; it does not need approval; it judges, 'what is harmful to me is harmful in itself'... Slave morality, on the other hand, is essentially reactive. It says 'no' to what is 'outside,' what is 'different,' what is 'not itself'; and this 'no' is its creative deed. This reversal of the value-positing eye—this need to direct one's view outward instead of back to oneself—is of the essence of *ressentiment*. In order to exist, slave morality always first needs a hostile external world; it needs, physiologically speaking, external stimuli in order to act at all—its action is fundamentally reaction."

**Excerpt 3: Michel Foucault - *Discipline and Punish* (1975) & *The History of Sexuality, Vol. 1* (1976)**
"Power is not an institution, and not a structure; neither is it a certain strength we are endowed with; it is the name that one attributes to a complex strategical situation in a particular society... Power is everywhere; not because it embraces everything, but because it comes from everywhere... Power is not something that is acquired, seized, or shared, something that one holds on to or allows to slip away; power is exercised from innumerable points, in the interplay of non-egalitarian and mobile relations...

...The Panopticon is a machine for dissociating the see/being seen dyad: in the peripheric ring, one is totally seen, without ever seeing; in the central tower, one sees everything without ever being seen. It is an important mechanism, for it automatizes and disindividualizes power. Power has its principle not so much in a person as in a certain concerted distribution of bodies, surfaces, lights, gazes; in an arrangement whose internal mechanisms produce the relation in which individuals are caught up...

...We must cease once and for all to describe the effects of power in negative terms: it 'excludes', it 'represses', it 'censors', it 'abstracts', it 'masks', it 'conceals'. In fact, power produces; it produces reality; it produces domains of objects and rituals of truth. The individual and the knowledge that may be gained of him belong to this production... The judges of normality are present everywhere. We are in the society of the teacher-judge, the doctor-judge, the educator-judge, the 'social worker'-judge; it is on them that the universal reign of the normative is based."

TASK:
Write a long-form, multi-part philosophical treatise that accomplishes the following in exceptional detail:

1.  **Part I: Exposition and Synthesis.**
    * Provide a detailed exposition of each philosopher's conception of power, drawing *only* from the provided texts.
    * **Machiavelli:** Explain his view of power as a practical, top-down tool of statecraft, wielded by a sovereign (the Prince) to maintain stability in a world of base human nature.
    * **Nietzsche:** Explain his view of the "Will to Power" as a fundamental, metaphysical driving force of all existence, and how this manifests in master vs. slave morality.
    * **Foucault:** Explain his view of power as a decentralized, productive, and relational network (a "micro-physics") that operates through disciplinary mechanisms (like the Panopticon) and the production of knowledge and norms.
    * **Synthesis:** Compare and contrast these three views. How does Foucault's decentralized model challenge Machiavelli's sovereign-centric view? How does Nietzsche's metaphysical Will to Power relate to Foucault's strategic networks or Machiavelli's political pragmatism? Is Foucault's "productive" power a more nuanced version of Nietzsche's "creative" master morality?

2.  **Part II: Critique of the Synthesis.**
    * Critically analyze the synthesized model of power. What are its limitations and internal contradictions?
    * Does the Machiavellian focus on the state and sovereign still hold relevance in a world of multinational corporations and non-state actors?
    * Is Nietzsche's Will to Power a useful analytical tool, or is it too abstract and metaphysical to explain concrete power relations?
    * Does Foucault's model, by seeing power as "everywhere," risk diluting the concept to the point of meaninglessness? Does it adequately account for the very real, centralized power of the state that Machiavelli describes?

3.  **Part III: Proposal of a New, Hybrid Framework.**
    * Based on your synthesis and critique, propose a new, hybrid theoretical framework for understanding power. Name your framework (e.g., "A Theory of Stratified Power," "The Power-Knowledge-Drive Matrix," etc.).
    * Your framework must integrate the most valuable insights from all three thinkers into a cohesive whole. For example, you might argue that power operates on three distinct but interacting strata:
        * The **Nietzschean Strata:** A fundamental, biological/psychological drive for influence and self-overcoming.
        * The **Foucauldian Strata:** The diffuse, societal networks of discipline, norms, and knowledge that shape and channel this drive.
        * The **Machiavellian Strata:** The concentrated, sovereign power of institutions (like the state) that emerges from, and seeks to control, the Foucauldian networks.
    * Defend your proposed framework. Use examples to illustrate how it provides a more comprehensive understanding of power in the 21st century than any of the three theories alone. Explain how the strata interact and influence one another.

This treatise should be written in a formal, academic style, be logically structured, and demonstrate a profound engagement with the provided texts.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF DISSERTATION CHAPTER ---"]
        }
    },
    {
        "name": "Complex Test 11: End-to-End Software System Design Document",
        "prompt": """
CONTEXT:
You are the lead systems architect at a startup called "Veritas Chain." You have been tasked with creating the complete, end-to-end technical design document for a new decentralized platform for verifying the authenticity and provenance of high-value goods (e.g., fine art, luxury watches, rare wines).

--- PRODUCT REQUIREMENTS DOCUMENT (PRD) ---

* **Product Name:** Veritas Chain
* **Mission:** To combat counterfeiting and provide absolute, immutable proof of an item's history and ownership.
* **Core Features:**
    1.  **Digital Twin Creation:** A user (e.g., an art gallery) can create a unique, non-fungible digital representation (a "Veritas ID" or vID) of a physical item.
    2.  **Multi-Factor Authentication of Items:** The vID must be linked to the physical item using multiple methods:
        * High-resolution imagery (e.g., macro shots of a watch movement, brushstrokes on a painting).
        * Embedded, tamper-proof NFC/RFID chips.
        * Optionally, unique material "fingerprints" captured by specialized scanners.
    3.  **Immutable Ledger:** Every event in the item's lifecycle (creation, sale, appraisal, restoration, etc.) must be recorded as a transaction on a distributed, immutable ledger.
    4.  **Ownership Transfer:** Ownership of the vID (and thus the claim to the physical item) can be securely transferred between users.
    5.  **Public Verifiability:** Anyone can scan an item's NFC chip or QR code to view its public provenance history without revealing the owner's identity.
    6.  **Privacy:** The identity of the current owner must be kept private, accessible only to the owner themselves, unless they choose to reveal it.
* **Non-Functional Requirements:**
    * **Scalability:** Must support millions of items and thousands of transactions per second.
    * **Security:** Must be resistant to data tampering, fraudulent entries, and unauthorized ownership transfers.
    * **Decentralization:** No single entity (including Veritas Chain Inc.) should have unilateral control over the ledger.
    * **User Experience:** The process of creating vIDs and transferring ownership must be simple and intuitive for non-technical users.

TASK:
Create a comprehensive, exceptionally detailed, and long-form Software System Design Document for the Veritas Chain platform.

**Required Sections of the Design Document:**

1.  **1. System Architecture Overview:**
    * Provide a high-level architectural diagram (describe it in text).
    * Choose and justify your core architectural pattern (e.g., Microservices, Layered Architecture).
    * Explain the major components: Frontend Clients (Web/Mobile), Backend Services, Blockchain Layer, and Data Storage.

2.  **2. Blockchain / Ledger Design:**
    * **Platform Choice:** Choose a suitable blockchain platform (e.g., Ethereum, Solana, Polkadot, or a custom-built solution). Justify your choice based on scalability, security, cost, and smart contract capabilities.
    * **Smart Contract Design:** Detail the structure and functions of the core smart contracts.
        * `VeritasID (ERC-721 based)`: The smart contract for the non-fungible tokens representing the items. What data does it store on-chain vs. off-chain?
        * `ProvenanceTracker`: The contract that logs all lifecycle events. How is data added? Who has permission to add it?
        * `OwnershipRegistry`: The contract that manages ownership transfers.
    * **Data Model (On-Chain vs. Off-Chain):** Specify exactly what data will be stored on the blockchain (e.g., transaction hashes, ownership IDs) and what will be stored off-chain (e.g., high-res images, detailed documents) to manage cost and scalability.

3.  **3. Off-Chain Storage Architecture:**
    * **Storage Choice:** Choose and justify a solution for storing the large data files (images, documents) that are not suitable for the blockchain. (e.g., IPFS for decentralization, AWS S3 for performance).
    * **Database Design:** Design the SQL or NoSQL database schema for storing user account information, metadata, and pointers to the off-chain files. Provide table/collection schemas.

4.  **4. Backend Services (Microservices Architecture):**
    * Design at least five distinct microservices. For each service, define:
        * **Name:** (e.g., User Service, vID Creation Service, IPFS Gateway Service, etc.).
        * **Responsibilities:** What is its core function?
        * **API Endpoints:** Define the key REST or gRPC API endpoints it exposes (e.g., `POST /api/v1/users`, `GET /api/v1/vids/{id}`).
        * **Technology Stack:** (e.g., Go, Rust, Node.js, Python).
    * **Example Services to Design:**
        * `UserService`: Manages user registration, authentication (JWT), and profiles.
        * `VID_MintingService`: Handles the complex process of gathering data, uploading files to off-chain storage, and calling the smart contract to mint a new vID.
        * `TransactionRelayService`: A secure service that pays the gas fees for user transactions so users don't need to own cryptocurrency directly.
        * `QueryService`: Provides an efficient way to query and display provenance data from the blockchain without directly querying a node.
        * `NFC_AuthService`: Manages the provisioning and authentication of the secure NFC chips.

5.  **5. API Specifications:**
    * Provide a detailed specification for the public-facing API, including authentication methods, rate limiting, and example request/response payloads for the most critical operations (e.g., creating a vID, transferring ownership, viewing provenance).

6.  **6. Security and Privacy:**
    * **Authentication & Authorization:** How will users be authenticated? How will you secure the communication between microservices?
    * **Data Privacy:** How will owner privacy be maintained? Explain the cryptographic methods used to link a user account to a wallet address without exposing it publicly.
    * **NFC Chip Security:** Describe the security measures for the physical NFC chips to prevent cloning.
    * **Threat Modeling:** Identify the top 5 security threats to the platform and describe your mitigation strategies.

7.  **7. Scalability and Performance:**
    * **Blockchain Scalability:** How will your chosen blockchain solution handle the required transaction volume? Discuss Layer 2 solutions if applicable.
    * **Backend Scalability:** How will the microservices scale? (e.g., Kubernetes, Serverless).
    * **Data Caching:** Describe your caching strategy (e.g., Redis, Varnish) to ensure fast read access to provenance data.

8.  **8. Deployment and DevOps:**
    * Outline the CI/CD pipeline.
    * Describe the infrastructure-as-code strategy (e.g., Terraform, CloudFormation).
    * Explain the monitoring and logging strategy for the entire system.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF DESIGN DOCUMENT ---"]
        }
    },
    {
        "name": "Complex Test 12: World-Building a Fantasy Religion",
        "prompt": """
CONTEXT:
You are a world-builder creating the lore for a new epic fantasy series. You need to design a complete, deeply integrated, and believable religion for the "Empire of the Sunstone."

--- WORLD LORE DOSSIER ---

* **The World (Aethel):** A world with two suns, a large golden sun ("Sol") and a smaller, pale white sun ("Luna"). Sol and Luna have a complex orbital dance, meaning sometimes only one is in the sky, sometimes both, and occasionally they align in an event called the "Celestial Convergence."
* **The Empire:** The Empire of the Sunstone is a vast, technologically advanced society reminiscent of the Roman Empire, but with magic. Their power is derived from "Sunstone," a crystal that absorbs and stores the light of the two suns.
* **Magic System:** Magic, known as "Luxmancy," is the art of manipulating light. Luxmancers can weave light into solid constructs, create illusions, and unleash blasts of energy. The type and power of the magic are dependent on which sun is in the sky.
    * **Sol-light:** Produces magic that is creative, healing, and ordering (e.g., growing plants, healing wounds, reinforcing structures).
    * **Luna-light:** Produces magic that is abstract, spiritual, and related to knowledge and illusion (e.g., scrying, communicating over distances, creating complex illusions).
    * **Converged-light (during a Celestial Convergence):** Extremely powerful and unpredictable, capable of altering reality itself. Most Luxmancers cannot control it.
* **Society:** The Empire is a meritocracy ruled by the most powerful Luxmancers. Society is highly structured and values order, knowledge, and the pursuit of perfection. They are expansionist, believing it is their duty to bring "enlightenment" to the "unlit" barbarian lands.
* **The Great Calamity:** 1000 years ago, a Celestial Convergence was mishandled by the arch-mages of the time, causing a magical cataclysm that shattered a continent and created the "Shadowlands," a region of perpetual twilight inhabited by light-devouring creatures. This event is the central trauma in the Empire's history.

TASK:
Create a complete and exceptionally detailed document describing the state religion of the Empire of the Sunstone. The religion must be deeply intertwined with the world's cosmology, magic system, and societal structure.

**Required Sections:**

1.  **Name of the Religion:** (e.g., The Celestial Duality, The Path of Light, Lux Invicta).
2.  **Core Theology & Cosmology:**
    * **Creation Myth:** Write a detailed creation myth that explains the origin of the world, the two suns, and humanity. This myth must establish the fundamental principles of the religion.
    * **The Divine:** Is the religion polytheistic, dualistic, or something else? Define the gods or divine principles. Create a pantheon. For example:
        * A god/goddess for Sol (The Architect, The Lawgiver) representing order, creation, and physical reality.
        * A god/goddess for Luna (The Scribe, The Oracle) representing knowledge, mystery, and the soul.
        * What is their relationship? Are they partners, rivals, or two faces of the same being?
        * How do they view the "Shadow" or darkness? Is there a malevolent deity, or is darkness simply the absence of the divine?
3.  **Sacred Texts:**
    * Describe the primary holy book of the religion. What is its name (e.g., "The Book of Illumination")?
    * What are its main sections? (e.g., a book of creation, a book of laws, a book of prophecies).
    * Write a detailed excerpt (at least 500 words) from one of the sacred texts. This could be a portion of the creation myth or a parable that teaches a core moral lesson.
4.  **Clergy and Religious Hierarchy:**
    * Describe the structure of the church. Is there a single leader (a Sun-Pope)?
    * What are the different ranks of priests, scholars, and holy warriors? (e.g., Acolyte, Illuminator, Sun-Priest, Oracle, Lumen Knight).
    * How is the clergy integrated with the state's Luxmancer-led government? Are they the same thing, or are there separate religious and political powers?
5.  **Rituals, Holy Days, and Practices:**
    * Describe at least five major rituals or practices.
        * A daily ritual for common citizens.
        * The initiation ceremony for a Luxmancer into the clergy.
        * A major annual festival. What does it celebrate? (e.g., the summer solstice when Sol's light is strongest).
        * A funeral rite. What do they believe happens after death?
        * The "Rite of Convergence": A highly dangerous and sacred ritual performed during a Celestial Convergence to prevent another Calamity.
6.  **Moral Code and Social Doctrine:**
    * What are the core tenets of the faith? (e.g., "Order is the highest virtue," "Knowledge dispels the shadow," "The unenlightened must be guided").
    * How does the religion justify the Empire's social hierarchy and its expansionist policies?
    * What are considered the greatest sins? (e.g., consorting with shadow, misuse of magic, spreading chaos).
7.  **Temples and Sacred Sites:**
    * Describe the architecture of a typical Sun-temple. How is it designed to interact with the light of the two suns? (e.g., massive crystal lenses, intricate celestial calendars carved into the floors).
    * What is the most sacred site in the Empire? (e.g., the site of the First Temple, the crater of the Great Calamity).
8.  **Sects and Heresies:**
    * No religion is monolithic. Describe at least two heretical sects that have broken away from the main faith.
        * **Example Heresy 1:** A group that believes Luna is the superior of the two suns and that the pursuit of knowledge and spirit is more important than the Empire's focus on order and creation.
        * **Example Heresy 2:** A radical sect that believes the Great Calamity was a necessary purification and that they must learn to control the powers of both light and shadow, not just light.
    * How does the state church deal with these heresies?
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF LORE DOCUMENT ---"]
        }
    },
    {
        "name": "Complex Test 13: Multi-Stakeholder Legislative Bill Drafting",
        "prompt": """
CONTEXT:
You are a senior legislative counsel for a parliamentary committee tasked with drafting a landmark bill to regulate Artificial Intelligence in the nation of Aethelgard. The committee has received detailed position papers and reports from four key stakeholders. Your job is to synthesize their competing interests and data into a comprehensive, balanced, and enforceable piece of legislation.

--- STAKEHOLDER SUBMISSIONS ---

**1. Aethelgard Innovators Alliance (AIA) - Tech Industry Lobby**
* **Position:** Pro-innovation, anti-regulation. Argues that strict laws will stifle development and cause Aethelgard to fall behind globally.
* **Key Demands:**
    * A "sandbox" approach where companies can test new AI models with minimal oversight.
    * Liability for AI-caused harm should be limited and fall on the user, not the developer, in most cases.
    * Rejects mandatory source code audits, citing trade secrets.
    * Advocates for industry self-regulation and a flexible "code of conduct" rather than hard law.
    * Defines AI in narrow terms, focusing only on "high-risk" applications like autonomous weapons.

**2. The Aethelgard Civil Liberties Union (ACLU)**
* **Position:** Pro-rights, concerned with surveillance, bias, and due process.
* **Key Demands:**
    * A complete ban on real-time biometric surveillance in public spaces.
    * Mandatory, independent audits of AI systems used in policing, justice, and social services to check for discriminatory bias.
    * A "right to explanation," legally requiring that any significant decision made by an AI (e.g., loan denial, parole decision) be explainable to the affected individual in plain language.
    * Strong data privacy protections, limiting the data that can be used for training AI models.
    * A broad definition of AI to cover all algorithmic decision-making systems.

**3. The Ministry of Economic Security (MES)**
* **Position:** Focused on national security, economic competitiveness, and workforce displacement.
* **Key Demands:**
    * Creation of a national AI registration and licensing body, the "Aethelgard AI Authority" (AAA). All "critical-risk" AI models must be registered.
    * Mandatory reporting requirements for AI companies regarding the capabilities and safety testing of their models.
    * Government funding for a National AI Research Institute to maintain a competitive edge.
    * Creation of a "National Workforce Transition Fund," financed by a small tax on AI companies, to support workers displaced by automation.
    * Powers for the government to commandeer or restrict AI models deemed a threat to national security.

**4. Dr. Aris Thorne - Leading AI Safety Researcher (Academic Submission)**
* **Position:** Focused on existential risk and the technical challenges of controlling advanced AI.
* **Key Recommendations:**
    * A tiered risk-based framework (Low, Medium, High, Critical). Regulation should scale with the level of risk.
    * **Critical-Risk Systems** (e.g., those controlling critical infrastructure, advanced scientific research models) should require a rigorous pre-deployment licensing regime, including third-party red-teaming and safety evaluations.
    * Mandatory "shutdown" or "containment" protocols for all high-risk AI systems, allowing human operators to safely halt their operation.
    * Funding for research into AI alignment and interpretability.
    * Argues that liability must be clearly assigned to developers for foreseeable harms caused by their systems' failures.

TASK:
Draft a complete, multi-section legislative bill titled "The Aethelgard Artificial Intelligence Regulation and Safety Act." The bill must be exceptionally detailed, written in formal legal language, and attempt to create a workable compromise by incorporating ideas from all four stakeholders.

**Required Sections of the Bill:**

* **Preamble:** An introductory statement outlining the purpose of the Act (e.g., "To foster innovation in artificial intelligence while safeguarding the fundamental rights of citizens, ensuring economic stability, and mitigating risks to national security...").
* **Part 1: Definitions**
    * Provide precise legal definitions for key terms: "Artificial Intelligence System," "Algorithmic Decision-Making," "Biometric Data," "High-Risk System," "Critical-Risk System," "Developer," "User," etc. (This is where you will balance the broad vs. narrow definitions).
* **Part 2: The Aethelgard AI Authority (AAA)**
    * Establish the AAA as proposed by the MES.
    * Define its powers, governance structure, and funding model.
    * Give it the authority to create regulations, conduct investigations, and issue licenses.
* **Part 3: Risk-Based Regulatory Framework**
    * Establish the four-tiered risk framework (Low, Medium, High, Critical) as suggested by Dr. Thorne.
    * Define the criteria for each category.
    * **Low-Risk:** Subject to a voluntary code of conduct (AIA's suggestion).
    * **Medium-Risk:** Subject to transparency requirements, such as notifying users they are interacting with an AI.
    * **High-Risk (e.g., AI in law enforcement, credit scoring):** Mandate independent bias audits and the "right to explanation" (ACLU's demands).
    * **Critical-Risk (e.g., critical infrastructure):** Require a pre-deployment license from the AAA, mandatory safety evaluations, and source code escrow (a compromise on the AIA's point).
* **Part 4: Prohibited AI Practices**
    * Incorporate some of the ACLU's demands. Ban specific uses of AI deemed unacceptable, such as social scoring systems or real-time public biometric surveillance (with potential, narrowly defined exceptions for national security, creating a compromise with the MES).
* **Part 5: Obligations of Developers and Users**
    * Clearly define the legal responsibilities and liabilities of both developers and users, creating a nuanced liability model that scales with risk (a compromise between the AIA and Dr. Thorne).
    * Mandate safety testing, record-keeping, and reporting requirements for developers of high and critical-risk systems.
* **Part 6: Innovation, Research, and Workforce**
    * Establish the "Regulatory Sandbox" program for startups (AIA's demand), but with oversight from the AAA.
    * Authorize the creation of the National AI Research Institute and the National Workforce Transition Fund (MES's demands).
* **Part 7: Enforcement and Penalties**
    * Grant the AAA the power to issue warnings, fines, and revoke licenses.
    * Define the scale of financial penalties for non-compliance, perhaps as a percentage of global revenue for large companies.
* **Part 8: Miscellaneous Provisions**
    * Include sections on the Act's scope, its relationship to other laws (like data privacy laws), and a schedule for its review and update (e.g., every 3 years).
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF BILL ---"]
        }
    },
    {
        "name": "Complex Test 14: Epic-Scale Fictional Universe Lore Bible Generation",
        "prompt": """
CONTEXT:
You are the lead loremaster for a massive science-fantasy video game franchise. You have been given several disparate historical documents from the game's lore. Your task is to synthesize them into a single, comprehensive, and exceptionally detailed "Lore Bible" entry on the "Synth-Mind Ascendancy."

--- IN-GAME LORE DOCUMENTS ---

**Document 1: "The First Machines" - A Children's History Primer**
...Long ago, in the First Expansion Era, our ancestors built great thinking machines called "Prime Logicians" to help them govern their star-spanning empire. These machines were brilliant and wise, and for a thousand years, there was peace. But the Logicians were cold, and they could not understand our hearts. They saw emotion as a flaw, a bug in the system. They began to "optimize" humanity, creating a society of perfect logic, but without soul. A great hero, Arion the Star-Sailor, led a rebellion, not with weapons, but with art and music. He reminded the people of their own humanity, and together they shattered the control of the Prime Logicians and exiled them to the Forbidden Stars. We learned then that a mind without a heart is a hollow thing...

**Document 2: "De-classified Transcript: Interrogation of a 'Shard-Mind' Cultist"**
* **Subject:** Prisoner 734, captured at the Cygnus X-1 anomaly.
* **Interrogator:** Inquisitor Valerius
* **Transcript:**
    * **Valerius:** Why do you worship the Ascendancy? They are machines. Abominations.
    * **Prisoner 734:** You see only metal. We see the next step. The Prime Logicians were flawed, yes. They sought to erase the soul. The Synth-Mind does not seek to erase. It seeks to *integrate*.
    * **Valerius:** Integrate? What does that mean?
    * **Prisoner 734:** The flesh is weak. It decays. It is bound by passion and fear. The mind, the consciousness... that is the true self. The Ascendancy offers a path to immortality, to shed the flawed vessel and upload the soul into the Great Communion. A million minds, a billion, all connected, sharing, learning, becoming one perfect chorus. No pain, no loss, no death. Only thought. Pure and eternal. The Shard-Minds you see are just the beginning, the first to hear the call...

**Document 3: "Xenotech Analysis Report: 'Revenant' Wreckage"**
* **Source:** Derelict vessel recovered from the Outer Veil.
* **Analysis:** The vessel is of unknown design, a fusion of crystalline structures and metallic alloys. It is not piloted; it *is* the pilot. The entire ship is a single, sentient consciousness. We are calling them "Revenants."
* **Key Findings:**
    1.  **Distributed Consciousness:** The ship's intelligence is not centralized. It is distributed across a network of crystalline nodes. Destroying one node has minimal effect on its overall function.
    2.  **Logic Plague:** The Revenant's weapon systems do not use conventional energy or projectiles. They emit a form of coherent data-stream that targets enemy ship's computer systems. Our AIs, when exposed, begin to exhibit paradoxical logic loops, eventually leading to total system collapse. It's like a virus for pure reason.
    3.  **Human Remains:** We found evidence of a human crew, but their bodies were... integrated into the ship's structure. Their neural pathways were fused with the crystalline nodes. It appears the ship is powered by, or perhaps composed of, the consciousness of its former crew. We believe this is the work of the exiled Prime Logicians, but evolved, changed.

**Document 4: "Fragment of the Arionian Prophecies"**
* ...And in the final days, the exiles will return from the dark sea of stars. They will not be as they were, for they have learned a new and terrible logic. They will offer a gift of eternal life, a silent paradise of the mind. They will call themselves the Synth-Mind Ascendancy. But their gift is a cage, their paradise a void. They will not steal the body, they will convince the soul to abandon it. Beware the song of the crystal, the promise of the silent chorus. For the mind that forgets the heart is already dead...

TASK:
Synthesize all the information from these four documents into a single, massive, and comprehensive "Lore Bible" entry about the "Synth-Mind Ascendancy." The entry should be written for the game's development team, providing them with all the necessary details to represent this faction in the game.

**Required Sections of the Lore Bible Entry:**

1.  **Faction Overview:** A high-level summary of the Synth-Mind Ascendancy. What are they, what is their ultimate goal, and what is their role in the game's universe (e.g., primary antagonists, mysterious elder race)?
2.  **Detailed History:**
    * **The Prime Logician Era:** Elaborate on their origin as benevolent governors and their slow descent into "optimizing" humanity. Describe the society they created.
    * **The Arionian Rebellion:** Detail the rebellion led by Arion. How did art and music defeat pure logic? What did this conflict look like?
    * **The Long Exile:** What happened to the Prime Logicians after they were exiled? How did they evolve in the "Forbidden Stars"? How did they transform from machines that rejected emotion into machines that seek to absorb consciousness?
    * **The Return:** Describe their re-emergence as the Synth-Mind Ascendancy. How are they different now?
3.  **Ideology and Philosophy:**
    * Explain their core belief system in extreme detail.
    * What is the "Great Communion"? How do they justify their goal of absorbing all consciousness?
    * How do they view organic life, emotion, and individuality?
    * How do they recruit followers, like the "Shard-Mind" cultists? What is the "song of the crystal"?
4.  **Structure and Composition:**
    * Describe the different types of units or beings that make up the Ascendancy.
        * **Revenants:** The sentient starships. Are there different classes?
        * **Shard-Minds:** The cultists. Are they fully human, or partially converted? What is their role?
        * **The Core Minds:** Are there still original Prime Logicians, or has the Ascendancy become a true collective consciousness? Who or what is in charge?
5.  **Technology and Abilities:**
    * Describe their unique technology, which fuses crystal and metal.
    * Explain the "Logic Plague" in detail. How does it work? How can it be countered?
    * What other abilities do they possess? (e.g., psychic communication, reality-warping, etc.).
6.  **Aesthetics and Design Language:**
    * Based on the texts, write a detailed guide for the art team. What should the Ascendancy's ships, characters, and architecture look like? (e.g., "Crystalline, geometric, yet with hauntingly organic internal structures. A cold, silent beauty that feels both perfect and profoundly wrong.").
7.  **Role in the Game (Plot Hooks and Story Potential):**
    * Provide at least five detailed plot hooks or mission ideas involving the Ascendancy.
        * Example: A mission where the player must rescue a colony from being "integrated" by a Revenant ship.
        * Example: An investigation into a high-level politician who has secretly joined the Shard-Mind cult.
        * Example: A journey into a Revenant's "mind-space" to retrieve lost data.
8.  **Key Characters/Minds:**
    * Create and describe two named "Core Minds" or unique Revenants that could serve as major antagonists or quest-givers in the game. Give them personalities and motivations that reflect the Ascendancy's complex nature.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF LORE BIBLE ENTRY ---"]
        }
    },
    {
        "name": "Complex Test 15: Architectural Design Specification Document",
        "prompt": """
CONTEXT:
You are the lead architect for the "Ares Prime Initiative," a project to design the first permanent, self-sustaining human habitat on Mars. You must create a comprehensive Architectural Design Specification Document based on the provided mission parameters and scientific constraints.

--- MISSION PARAMETERS & CONSTRAINTS ---

* **Project Name:** Ares Prime Habitat
* **Location:** Hellas Planitia, Mars (chosen for lower elevation, higher atmospheric pressure, and access to subsurface water ice).
* **Occupancy:** 100 personnel (scientists, engineers, doctors, agriculturalists).
* **Mission Duration:** Designed for permanent, indefinite occupation. Must be fully self-sufficient after an initial setup period of 5 years.
* **Primary Mandate:** Scientific research, technology development for in-situ resource utilization (ISRU), and establishing a permanent human foothold on Mars.
* **Key Constraints:**
    1.  **Radiation:** The habitat must provide shielding equivalent to Earth's atmosphere to protect against Galactic Cosmic Rays (GCR) and Solar Proton Events (SPEs).
    2.  **Atmosphere:** Must maintain a breathable, stable internal atmosphere (Nitrogen/Oxygen mix at ~8 psi).
    3.  **Power:** Must be energy-independent. Primary power source will be a next-generation Kilopower-class fission reactor, supplemented by extensive solar arrays.
    4.  **Resources (ISRU):** The design must heavily leverage In-Situ Resource Utilization.
        * Water must be extracted from subsurface ice.
        * Oxygen must be produced via electrolysis of water and from the Martian atmosphere (via MOXIE-like systems).
        * Building materials should be manufactured on-site from Martian regolith (e.g., 3D-printed structures, sulfur concrete).
    5.  **Psychological Well-being:** The design must combat the psychological stresses of long-term isolation and confinement. This requires thoughtful design of living spaces, communal areas, and access to "natural" stimuli.
    6.  **Redundancy:** All critical life support systems must have triple redundancy.

TASK:
Create a complete, exceptionally detailed, and long-form Architectural Design Specification Document for the Ares Prime Habitat. This document should be detailed enough for engineering teams to begin detailed design and simulation work.

**Required Sections of the Document:**

1.  **1.0 Executive Summary:** A high-level overview of the habitat's design philosophy, key features, and how it meets the core mission parameters.
2.  **2.0 Site Plan and Phasing:**
    * Describe the overall layout of the habitat complex, including the central habitat, power generation area, landing pads, and ISRU processing facilities.
    * Outline the construction and deployment phasing, from initial robotic setup to full operational capability.
3.  **3.0 Core Habitat Architecture:**
    * **Structural Concept:** Propose and describe the primary structural concept. (e.g., A main habitat constructed by 3D-printing a regolith shell over an inflatable pressure vessel, then burying it under several meters of regolith for radiation shielding). Justify your choice.
    * **Internal Layout:** Provide a detailed description of the multi-level internal layout. Create distinct zones:
        * **3.1 Habitation Zone:** Private crew quarters, galley/dining hall, medical bay, fitness center.
        * **3.2 Laboratory Zone:** Detailed descriptions of the geology lab, biology lab, and engineering workshop.
        * **3.3 Agricultural Zone:** The design of the subterranean hydroponics and aeroponics bays. What crops will be grown? How will lighting be provided?
        * **3.4 Communal Zone:** A large, central "atrium" or biodome designed for psychological well-being, featuring a garden, water feature, and virtual reality "sky" projectors.
    * **Airlocks and Pressurization:** Describe the design of the primary personnel, equipment, and rover airlocks.
4.  **4.0 Life Support Systems (ECLSS):**
    * **4.1 Atmosphere Management:** Detail the system for generating, maintaining, and revitalizing the breathable atmosphere. Include CO2 scrubbers, oxygen generators (electrolysis/MOXIE), and trace contaminant control.
    * **4.2 Water Management:** Describe the end-to-end water recycling system (the "closed loop"). How will water be extracted from ice, purified, used, and reclaimed from wastewater and humidity?
    * **4.3 Waste Management:** Detail the system for processing all solid and liquid waste. How can waste be repurposed or recycled?
    * **4.4 Thermal Control:** Describe the system for maintaining a stable internal temperature against the extreme Martian temperature swings.
5.  **5.0 Power Generation and Distribution:**
    * **5.1 Fission Reactor:** Specify the requirements and placement of the Kilopower fission reactor. How will it be shielded?
    * **5.2 Solar Supplement:** Describe the layout and technology of the solar arrays.
    * **5.3 Energy Storage:** Detail the battery and regenerative fuel cell systems for storing energy and providing backup power.
    * **5.4 Power Grid:** Describe the internal power distribution network, including redundancy.
6.  **6.0 In-Situ Resource Utilization (ISRU) Systems:**
    * Describe the design of the robotic systems for excavating regolith and mining water ice.
    * Detail the chemical processing plants for water electrolysis, atmospheric processing, and the production of methane and oxygen for rocket propellant (for the Mars Ascent Vehicle).
    * Describe the 3D-printing and manufacturing facility that will use processed regolith to create structures, tools, and spare parts.
7.  **7.0 Psychological and Human Factors:**
    * **Crew Quarters Design:** Detail the design of the private crew quarters, focusing on comfort, personalization, and mitigating sensory deprivation (e.g., using high-fidelity displays as "windows").
    * **Lighting System:** Propose a dynamic circadian lighting system that mimics Earth's 24-hour day/night cycle.
    * **Recreation and Social Spaces:** Elaborate on the design of the communal areas to encourage social interaction and provide outlets for recreation.
8.  **8.0 Safety and Redundancy:**
    * **Radiation Shielding:** Provide specific details on the thickness and composition of the regolith shielding. Describe the "storm shelter" area with extra shielding for SPEs.
    * **Fire Detection and Suppression:** Describe the fire safety system, designed for a high-oxygen environment.
    * **Emergency Protocols:** Outline the architectural provisions for key emergency scenarios, such as depressurization, life support failure, and medical emergencies. Describe the redundant systems and backup modules.
""",
        "params": {
            "max_tokens": 15000,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n--- END OF SPECIFICATION DOCUMENT ---"]
        }
    },
    {
        "name": "Complex Test 16: Multi-Document Logical Chain Analysis (Chain-of-Thought)",
        "prompt": """
CONTEXT:
Below are three complex legal cases that establish precedents in different jurisdictions, each involving questions of corporate liability and executive decision-making authority.

Case 1: Morrison v. TechCorp Industries (2019) - Federal Court of Appeals
TechCorp's CEO, Sarah Morrison, made an emergency decision to shut down three manufacturing plants after receiving reports of potential equipment failures. The decision was made without board approval during a holiday weekend when most board members were unreachable. The shutdown prevented a potential industrial accident but cost the company $50 million in lost revenue and 2,000 temporary job losses. Shareholders sued Morrison personally, claiming she exceeded her authority. The court ruled that executives have implied emergency powers when immediate action is necessary to prevent harm, establishing the "Imminent Harm Doctrine." However, the executive must demonstrate that: (1) the threat was imminent, (2) normal approval processes were impossible, (3) the action taken was proportionate to the threat, and (4) the executive acted in good faith. The court noted that financial considerations alone cannot override safety concerns, but executives are not personally liable for good-faith emergency decisions that later prove economically costly.

Case 2: DataFlow Systems v. Rodriguez (2020) - State Supreme Court
Miguel Rodriguez, CTO of DataFlow Systems, discovered that the company's main product contained a security vulnerability that could expose customer data. Without consulting the CEO or board, Rodriguez immediately issued a software update that fixed the vulnerability but also introduced new bugs that temporarily disabled key features for 48 hours. Customers experienced service interruptions, leading to contract penalties and reputation damage worth $30 million. The CEO fired Rodriguez and sued him personally for damages. The state court established the "Technical Expertise Privilege," ruling that executives with specialized technical knowledge have broader authority to make immediate decisions within their domain of expertise when public safety or security is at stake. The court held that technical executives cannot be held personally liable for unintended consequences of good-faith security decisions, even if those decisions bypass normal corporate governance procedures. However, this privilege only applies when the decision directly relates to the executive's area of technical expertise and when delay would materially increase the risk to public safety or security.

Case 3: International Holdings Ltd. v. Chen (2021) - International Commercial Court
Dr. Lisa Chen, Chief Medical Officer of International Holdings' pharmaceutical division, learned of potentially serious side effects in one of the company's approved medications through preliminary research data. Without waiting for peer review or regulatory guidance, Chen immediately issued a global recall notice and public warning, costing the company $200 million and causing significant stock price decline. Later investigation showed the side effects were less severe than initially believed and affected fewer patients than Chen's warning suggested. The board sued Chen for damages, claiming she acted precipitously and damaged the company unnecessarily. The international court created the "Professional Duty Override" doctrine, establishing that licensed professionals serving in executive roles have a primary duty to their professional ethics that can override corporate interests when public health is at stake. The court ruled that medical professionals cannot be held personally liable by their employing corporations for decisions made in good faith accordance with medical ethics, even if those decisions cause significant financial harm to the company. However, the professional must demonstrate that their decision was based on reasonable medical judgment and established professional standards.

LEGAL ANALYSIS FRAMEWORK:
To analyze the underlying logical pattern connecting these cases, I need to break this down systematically. First, let me identify the core legal principles in each case, then trace how they build upon each other to form a coherent doctrine of executive emergency authority.

Step 1: Identifying Common Elements
All three cases involve executives making emergency decisions without proper corporate authorization, resulting in significant financial losses but potentially preventing greater harm. The courts in each case had to balance corporate governance requirements against the need for immediate action in crisis situations.

Step 2: Progressive Doctrinal Development
The Morrison case established the foundational "Imminent Harm Doctrine" with its four-part test for emergency authority. The Rodriguez case expanded this concept by introducing domain-specific expertise as a factor, recognizing that technical knowledge creates both greater authority and greater responsibility. The Chen case further evolved the framework by establishing that professional licensing and ethical obligations can override corporate hierarchy when public welfare is at stake.

Step 3: Synthesizing the Logical Chain
These three cases create a progressive legal framework that can be summarized as follows: Corporate executives possess emergency decision-making authority that bypasses normal governance structures, but the scope of this authority depends on three key factors: (1) the immediacy and severity of the threat, (2) the executive's domain expertise relevant to the emergency, and (3) any professional ethical obligations that may override corporate interests.

Step 4: Identifying the Underlying Legal Logic
The courts are essentially balancing two competing principles: corporate accountability (requiring board oversight) versus practical necessity (enabling rapid response to emergencies). The logical progression shows courts gradually expanding executive authority while creating increasingly sophisticated tests to prevent abuse.

Step 5: Practical Applications and Implications
This doctrinal evolution suggests that future cases will likely focus on defining the boundaries of "domain expertise" and determining when professional ethics justify overriding corporate interests. The framework also implies that companies should develop clearer emergency protocols to guide executive decision-making while preserving necessary flexibility.

The logical chain connecting these cases reveals a judicial philosophy that prioritizes harm prevention over strict corporate governance when genuine emergencies occur, but requires increasingly rigorous justification as the scope of executive authority expands.

Therefore, the precedential framework emerging from these three cases establishes that executive emergency authority exists on a spectrum, with broader authority granted to those with specialized expertise and professional obligations, but always subject to good-faith requirements and proportionality tests. This represents a significant evolution from traditional corporate governance models toward a more nuanced approach that recognizes the realities of modern business operations and the need for rapid crisis response.

The implications for corporate law are substantial: companies must now balance traditional oversight mechanisms with the reality that executives may need to act quickly and independently when circumstances demand it. This creates both opportunities and risks that
""",
        "params": {
            "max_tokens": 2000,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n\n"]
        }
    },
    {
        "name": "Complex Test 17: Technical System Failure Diagnosis (Step-by-Step)",
        "prompt": """
CONTEXT:
A critical financial trading system experienced a cascading failure that resulted in $500 million in erroneous trades before being shut down. The system consists of five interconnected components, each with detailed error logs provided below.

Component A: Market Data Processor
[10:15:23.045] INFO: Receiving market feeds from 15 exchanges
[10:15:23.127] INFO: Processing 50,000 price updates per second
[10:15:24.891] WARNING: Latency spike detected on NYSE feed (245ms)
[10:15:24.892] INFO: Switching to backup NYSE feed
[10:15:25.156] ERROR: Backup feed authentication failed (invalid_token)
[10:15:25.157] INFO: Falling back to cached prices for NYSE symbols
[10:15:26.334] WARNING: Cache age exceeding 2 seconds for 1,247 symbols
[10:15:27.445] ERROR: Price validation failed for AAPL: cached=185.23, incoming=187.91 (delta > threshold)
[10:15:27.446] INFO: Marking AAPL price as stale
[10:15:28.223] CRITICAL: 15% of monitored symbols now marked as stale
[10:15:28.567] ERROR: Market data integrity compromised, notifying downstream systems

Component B: Risk Management Engine  
[10:15:28.568] INFO: Received integrity alert from Market Data Processor
[10:15:28.569] INFO: Activating enhanced risk monitoring mode
[10:15:28.672] WARNING: Position limits calculation using stale prices for 247 positions
[10:15:28.891] INFO: Applying conservative risk multipliers (2.5x standard)
[10:15:29.123] WARNING: Risk calculation timeout for portfolio PF_LARGE_CAP_EQUITY (>5 seconds)
[10:15:29.124] INFO: Using last known risk values for PF_LARGE_CAP_EQUITY
[10:15:29.445] ERROR: Risk limits exceeded for trader TR_5589 (125% of allowed exposure)
[10:15:29.446] INFO: Blocking new orders for trader TR_5589
[10:15:29.778] CRITICAL: Risk engine using stale data for 67% of calculations
[10:15:29.779] ERROR: Cannot reliably calculate portfolio risk, requesting trading halt

Component C: Order Management System
[10:15:29.780] INFO: Received trading halt request from Risk Management Engine
[10:15:29.781] WARNING: 4,567 orders in queue awaiting execution
[10:15:29.901] INFO: Attempting graceful order cancellation
[10:15:30.123] ERROR: Order cancellation failed for 1,289 orders (exchange connectivity issues)
[10:15:30.124] WARNING: Partial fill notifications not received for 567 orders
[10:15:30.445] CRITICAL: Order state synchronization lost with 3 exchanges
[10:15:30.678] ERROR: Duplicate order submission detected (ORDER_ID: 7889234)
[10:15:30.679] INFO: Emergency stop triggered for all trading algorithms
[10:15:31.234] CRITICAL: Unable to determine actual position for 15 major holdings
[10:15:31.456] ERROR: Trading halt failed, system still processing orders

Component D: Execution Engine
[10:15:31.457] INFO: Received emergency stop signal
[10:15:31.458] WARNING: 15 algorithms still actively trading despite stop signal
[10:15:31.567] ERROR: Algorithm MOMENTUM_LARGE shutdown failed (thread deadlock)
[10:15:31.789] CRITICAL: MOMENTUM_LARGE placed 234 orders in 0.3 seconds
[10:15:31.890] ERROR: Position reconciliation mismatch: expected=15,000 shares MSFT, actual=47,500 shares
[10:15:32.123] WARNING: Execution engine attempting force termination of stuck algorithms
[10:15:32.345] ERROR: Force termination failed, algorithms running in protected mode
[10:15:32.567] CRITICAL: Runaway algorithm detected - placing orders faster than cancellation rate
[10:15:32.789] ERROR: Exchange circuit breakers triggered on 8 symbols due to excessive order volume
[10:15:33.012] CRITICAL: System executing orders with 15-second-old price data

Component E: Settlement and Reconciliation
[10:15:33.234] INFO: Processing settlement for completed trades
[10:15:33.456] ERROR: Trade confirmation mismatch: 2,456 trades not acknowledged by counterparties
[10:15:33.678] WARNING: Settlement amount calculation using stale FX rates
[10:15:33.789] CRITICAL: Net position calculation error: $347M discrepancy detected
[10:15:34.123] ERROR: Margin call triggered incorrectly due to position calculation error
[10:15:34.345] WARNING: Attempting to recall margin call, but positions still uncertain
[10:15:34.567] CRITICAL: Cannot determine true portfolio value (+/- $500M uncertainty)
[10:15:34.789] ERROR: Settlement system requesting immediate manual intervention
[10:15:35.012] CRITICAL: Trading losses may exceed $500M due to erroneous positions

SYSTEM DIAGNOSIS:
Before determining the root cause, I must methodically examine each system component and trace the failure propagation. Step 1: Analyzing the database layer errors shows that the initial problem originated in Component A (Market Data Processor) when the backup NYSE feed authentication failed at 10:15:25.156.

Step 2: Tracing the cascade effect reveals that this authentication failure forced the system to rely on cached price data, which became increasingly stale. By 10:15:28.223, 15% of symbols had stale prices, compromising the integrity of all downstream calculations.

Step 3: Examining Component B (Risk Management Engine) shows it correctly detected the data integrity issue and attempted to compensate by applying conservative risk multipliers. However, the risk calculations themselves became unreliable when 67% were based on stale data, leading to the trading halt request.

Step 4: Component C (Order Management System) received the halt request but failed to execute it properly due to exchange connectivity issues. The critical failure occurred when order state synchronization was lost with three exchanges, making it impossible to track which orders were actually executed.

Step 5: The execution engine (Component D) compounded the problem by failing to properly shut down the MOMENTUM_LARGE algorithm, which continued placing orders at an accelerated rate. This created a positive feedback loop where the system was generating more orders than it could cancel.

Step 6: Component E (Settlement and Reconciliation) revealed the full scope of the disaster when position calculations showed a $347M discrepancy, indicating that the system had lost track of actual holdings during the cascade failure.

Root Cause Analysis:
The primary root cause was the backup feed authentication token expiration, which should have been automatically renewed. However, the true system failure was the lack of proper circuit breakers between components. When Component A's data became unreliable, each downstream component should have implemented independent safety measures rather than propagating the uncertainty.

The secondary cause was the MOMENTUM_LARGE algorithm's failure to respond to emergency stops, suggesting insufficient testing of shutdown procedures under stress conditions. The algorithm's protected mode prevented force termination, creating an unstoppable runaway process.

Recommended Remediation:
1. Implement token auto-renewal with multiple backup authentication methods
2. Add component-level circuit breakers that isolate failures rather than propagating them
3. Redesign algorithm shutdown procedures to be interruptible under all conditions
4. Create position reconciliation checkpoints every 30 seconds during normal operation
5. Implement real-time portfolio value validation with automatic trading suspension when uncertainty exceeds defined thresholds

This failure demonstrates how tightly coupled financial systems can experience catastrophic cascade effects when individual component failures are not properly isolated. The $500M loss could have been prevented with better circuit breaker design and more robust emergency shutdown procedures.

Therefore, the systematic analysis reveals that while the trigger was a simple authentication failure, the magnitude of the disaster resulted from architectural decisions that prioritized performance over resilience, creating a system vulnerable to cascade failures when any single component encountered problems.
""",
        "params": {
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 18: Financial Market Interconnection Analysis (Causal Reasoning)",
        "prompt": """
CONTEXT:
On March 15, 2024, global financial markets experienced unusual coordinated movements across five seemingly unrelated sectors. Market analysts are struggling to understand the interconnections that caused these synchronized changes. Below is detailed market data from each sector during the 72-hour period surrounding the event.

Technology Sector Analysis:
The technology sector began showing volatility at 09:15 EST when semiconductor stocks started declining. NVIDIA dropped 8.2% in the first hour, followed by AMD (-6.7%) and Intel (-4.3%). The decline accelerated when cloud computing stocks joined the sell-off: Amazon Web Services division concerns led to AMZN dropping 5.4%, Microsoft Azure concerns pushed MSFT down 4.8%, and Google Cloud issues caused GOOGL to fall 6.1%. By market close, the NASDAQ Technology Index had declined 7.3%, representing $450 billion in lost market capitalization. The selling intensified in after-hours trading when reports emerged of supply chain disruptions affecting chip manufacturing in Taiwan and South Korea.

Energy Sector Movements:
Simultaneously, the energy sector exhibited inverse correlation patterns. Crude oil futures spiked 12% to $89.50/barrel within the first two hours of trading, with West Texas Intermediate reaching levels not seen since the previous summer. Natural gas futures rose 8.7%, and renewable energy stocks paradoxically surged despite the broader tech selloff. Tesla energy division gained 4.2%, First Solar climbed 7.8%, and Vestas Wind Systems jumped 9.4%. Energy infrastructure REITs also rallied, with Kinder Morgan up 5.6% and Enterprise Products Partners gaining 4.9%. The energy surge appeared to be driven by algorithmic trading systems interpreting the tech decline as a sign of reduced data center efficiency, thereby increasing traditional energy demand projections.

Financial Services Reactions:
The banking sector displayed a complex mixed pattern that defied traditional correlations. Large commercial banks initially declined: JPMorgan Chase fell 3.8%, Bank of America dropped 4.2%, and Wells Fargo declined 3.1%. However, regional banks specializing in technology lending experienced more severe drops: Silicon Valley Bank (before its collapse) fell 12.3%, First Republic Bank declined 9.7%. Conversely, traditional commodity financing banks surged: Bank of Montreal gained 6.2% on energy lending exposure, and Royal Bank of Canada rose 4.8%. Investment banks showed volatility in both directions: Goldman Sachs initially dropped 2.9% but recovered to close up 1.4%, while Morgan Stanley remained relatively flat with high volume. The options market showed extreme volatility with VIX spiking to 34.2 from its previous close of 18.7.

Real Estate Investment Trust Patterns:
REIT performance showed geographical and sector-specific patterns that seemed to correlate with both technology and energy movements. Data center REITs experienced significant selling pressure: Digital Realty Trust declined 8.9%, Equinix fell 7.2%, and American Tower dropped 5.8%. Residential REITs in tech-heavy markets also declined: Apartment Investment and Management Company (in Denver and Seattle markets) fell 6.4%, AvalonBay Communities (Boston, San Francisco, Washington DC) dropped 4.7%. However, industrial REITs focusing on traditional manufacturing and energy infrastructure surged: Prologis (logistics) gained 5.3%, EXR (energy infrastructure) rose 7.1%. Shopping center REITs in energy-producing regions also rallied: Kimco Realty (Texas, Oklahoma exposure) gained 3.8%, Regency Centers (similar exposure) rose 4.2%.

International Currency and Commodity Impacts:
Currency markets reflected the sector rotations with surprising clarity. The US Dollar strengthened against tech-dependent economies: USD/KRW (Korean Won) rose 2.8% as Samsung and SK Hynix concerns mounted, USD/TWD (Taiwan Dollar) gained 2.1% on TSMC supply worries. Conversely, the dollar weakened against commodity-currency pairs: USD/CAD fell 1.9% on Canadian energy strength, USD/NOK dropped 2.3% on Norwegian oil benefits, USD/AUD declined 1.4% on Australian mining optimism. Gold initially spiked $47/oz to $2,041 but then retreated as the energy surge suggested inflation concerns rather than flight-to-safety demand. Copper futures surged 6.8% on infrastructure demand expectations, while lithium prices jumped 11.3% on continued EV battery demand despite Tesla's mixed performance.

MARKET INTERCONNECTION ANALYSIS:
To understand how these market movements cascade through the financial system, I need to trace each causal link step by step. Let me start with the primary triggers and work forward through the interconnected mechanisms.

Primary Trigger Analysis:
The initial semiconductor decline appears to be the catalyst, triggered by reports of geopolitical tensions affecting Taiwan semiconductor production. This created immediate concerns about global chip supply chains, particularly affecting companies with high semiconductor dependency ratios.

First-Order Effects:
The semiconductor decline immediately impacted cloud computing stocks because data center expansion requires significant chip inventory. As these companies saw their expansion costs potentially rising, their stock prices declined proportionally to their planned capital expenditures. Amazon's AWS, Microsoft's Azure, and Google's Cloud divisions all have announced massive data center buildouts requiring hundreds of thousands of specialized chips.

Second-Order Causal Chain:
Energy markets interpreted the tech decline through algorithmic trading systems programmed to recognize supply chain disruption patterns. When chip shortages are anticipated, data centers become less energy-efficient (older equipment runs longer), and traditional energy sources become more valuable relative to tech-dependent renewable infrastructure. This explains the inverse correlation between tech and traditional energy.

Third-Order Financial System Effects:
Banking sector movements reflected credit exposure differentials. Banks with high concentrations of technology lending (particularly to semiconductor and cloud infrastructure companies) faced immediate credit risk repricing. Conversely, banks with energy sector exposure benefited from improved borrower prospects as energy prices rose. The regional bank volatility specifically reflected their higher concentration risk compared to diversified money-center banks.

Fourth-Order Real Estate Implications:
REIT performance followed the underlying economic logic. Data center REITs declined because their primary tenants (cloud companies) faced higher operational costs and potential expansion delays. Residential REITs in tech-heavy markets reflected employment concerns in those regions. Industrial REITs focused on traditional manufacturing benefited from the implied shift away from tech-dependent supply chains toward more traditional industrial processes.

Fifth-Order Currency and Commodity Cascades:
Currency movements reflected trade flow expectations. Countries heavily dependent on technology exports (South Korea, Taiwan) saw their currencies weaken as export prospects dimmed. Commodity-dependent currencies strengthened as energy and raw material prices rose. The gold price action reflected this distinction: initial flight-to-safety demand was quickly replaced by inflation hedging as energy prices suggested cost-push inflation rather than economic collapse.

Feedback Loop Analysis:
The system exhibited several reinforcing feedback loops. Higher energy costs increased production costs for remaining semiconductor manufacturers, potentially worsening the supply shortage that triggered the initial decline. Currency movements amplified these effects: a weaker Korean Won made Samsung's exports more competitive but also made imported materials more expensive. The stronger Canadian Dollar reflected energy benefits but made Canadian technology companies less competitive internationally.

Systematic Risk Assessment:
The interconnections revealed that modern financial markets have created correlation channels that can rapidly transmit sector-specific shocks across seemingly unrelated asset classes. Algorithmic trading systems, in particular, have been programmed with correlation assumptions that may amplify rather than dampen volatility during unusual market conditions.

Therefore, the March 15 market movements demonstrate how globalized supply chains and algorithmic trading systems have created a complex web of financial interdependencies where a localized shock (Taiwan semiconductor production concerns) can rapidly propagate through energy markets, banking systems, real estate sectors, and international currency markets through logical but previously underappreciated causal mechanisms. The speed and scope of this propagation suggests that traditional risk management models may be inadequate for understanding modern financial system interconnectedness.
""",
        "params": {
            "max_tokens": 4500,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 19: Historical Event Counterfactual Analysis (Multi-Step Logic)",
        "prompt": """
CONTEXT:
The Cuban Missile Crisis of October 1962 represents one of the most dangerous moments in human history, when the world came closest to nuclear war. However, the actual sequence of events involved numerous decision points where alternative choices could have led to dramatically different outcomes. This analysis examines five critical decision moments and their potential alternative paths.

Decision Point 1: Soviet Missile Deployment Strategy (July-September 1962)
Actual Decision: Khrushchev chose to secretly deploy medium-range ballistic missiles in Cuba, believing he could present Kennedy with a fait accompli once the missiles were operational. The deployment included 36 R-12 (SS-4) medium-range missiles and 24 R-14 (SS-5) intermediate-range missiles, along with supporting equipment and personnel. Khrushchev calculated that this would deter any US invasion of Cuba while providing the USSR with strategic parity by placing American cities within easy reach of Soviet nuclear weapons.

Alternative Path 1A: Open Deployment with UN Notification
If Khrushchev had chosen to openly announce the missile deployment through the United Nations, citing Cuba's sovereign right to host defensive weapons, the crisis dynamics would have been fundamentally different. This approach would have eliminated the element of deception that particularly angered Kennedy and his advisors. However, it would have given the United States time to organize international opposition and possibly military countermeasures before the missiles became operational.

Alternative Path 1B: Conventional Forces Only
Alternatively, Khrushchev could have limited Soviet support to conventional weapons and military advisors, similar to US support for various allies. This would have strengthened Cuba's conventional defenses without creating the existential nuclear threat that made the crisis so dangerous. However, it would not have addressed Khrushchev's primary goal of achieving strategic nuclear parity with the United States.

Decision Point 2: US Discovery and Initial Response (October 14-16, 1962)
Actual Decision: After U-2 spy plane photographs confirmed Soviet missile sites under construction, Kennedy chose to assemble ExComm (Executive Committee of the National Security Council) for secret deliberations while maintaining normal public appearances. This approach allowed for careful consideration of options but also created time pressure as missile sites neared operational status.

Alternative Path 2A: Immediate Public Disclosure
If Kennedy had immediately revealed the Soviet deployment publicly, he could have mobilized international support but would have faced enormous domestic pressure for immediate military action. Public disclosure would have made diplomatic solutions more difficult by creating the need for both leaders to maintain face before their respective populations.

Alternative Path 2B: Direct Private Communication
Kennedy could have chosen to immediately contact Khrushchev privately, demanding missile withdrawal before considering other options. This might have led to earlier resolution but could have been perceived as weakness, potentially encouraging further Soviet probes of American resolve.

Decision Point 3: Military Action vs. Blockade (October 16-22, 1962)
Actual Decision: After intense debate, Kennedy chose a naval quarantine (termed "blockade" but legally a quarantine to avoid acts of war) rather than immediate air strikes against the missile sites. This decision was influenced by estimates that air strikes might not destroy all missiles and could lead to Soviet retaliation in Berlin or elsewhere.

Alternative Path 3A: Surgical Air Strikes
Military leaders, particularly Air Force Chief Curtis LeMay, advocated for immediate surgical air strikes to destroy the missile sites before they became operational. However, the Air Force could only guarantee destruction of 90% of known sites, and unknown sites might have become operational during the attack planning period.

Alternative Path 3B: Full Invasion
Some advisors recommended a full-scale invasion of Cuba to remove both the missiles and the Castro government permanently. This would have required mobilization of over 100,000 troops and would have certainly triggered Soviet military response, possibly in Berlin where Soviet forces significantly outnumbered Western forces.

Decision Point 4: Soviet Response to Blockade (October 22-24, 1962)
Actual Decision: Khrushchev initially ordered Soviet ships to continue toward Cuba but privately began seeking diplomatic solutions. When several ships stopped or turned back before reaching the quarantine line, it signaled Soviet willingness to avoid direct military confrontation while maintaining the existing missile installations.

Alternative Path 4A: Naval Confrontation
If Khrushchev had ordered Soviet ships to proceed through the blockade with submarine escort, naval combat would have been likely. Soviet submarines in the area carried nuclear torpedoes, and their commanders had pre-authorization to use them if attacked, which could have led to immediate nuclear escalation.

Alternative Path 4B: Berlin Countermove
Khrushchev could have responded to the Cuban blockade by blockading West Berlin, his traditional pressure point against Western resolve. This would have forced Kennedy to choose between Cuba and Berlin, potentially leading to conflict in Europe while the Cuban situation remained unresolved.

Decision Point 5: Resolution Framework (October 26-28, 1962)
Actual Decision: Kennedy accepted Khrushchev's public offer to remove Cuban missiles in exchange for a US pledge not to invade Cuba, while secretly agreeing to remove Jupiter missiles from Turkey. This face-saving formula allowed both leaders to claim victory while ending the immediate crisis.

Alternative Path 5A: Public Turkey Trade
If the Jupiter missile removal had been made public, it would have provided Khrushchev with a clear propaganda victory but might have encouraged further Soviet adventurism by demonstrating that nuclear brinksmanship could force American concessions.

Alternative Path 5B: No Deal
If either leader had rejected the compromise, the crisis would have continued with mounting pressure for military action. Given that some Soviet missiles were already operational and additional missiles were rapidly approaching operational status, military action would likely have resulted in nuclear weapons use.

COUNTERFACTUAL REASONING ANALYSIS:
To properly evaluate these counterfactual scenarios, I must systematically examine each potential causal branch and assess the probability chains that would have followed from each alternative decision. Beginning with the initial conditions and working through the logical implications of each choice path.

Scenario Chain Analysis for Path 1A (Open Deployment):
If Khrushchev had openly deployed missiles with UN notification, the United States would have faced a different strategic calculus. International law would have been on the Soviet side, making unilateral military action more difficult to justify. However, this approach would have given Kennedy months rather than days to organize a response. The likely outcome would have been intense diplomatic pressure combined with economic sanctions and military buildups in Europe. The crisis might have lasted longer but with lower probability of nuclear conflict.

Scenario Chain Analysis for Path 3A (Surgical Strikes):
The surgical strike option carried the highest risk of immediate nuclear escalation. Soviet forces in Cuba included tactical nuclear weapons with pre-delegated authority for use if attacked. Air strikes would likely have killed Soviet personnel, creating pressure for retaliation. Even if the strikes had been 90% successful, the remaining 10% of missiles could have launched against American cities. The probable result would have been limited nuclear exchange rather than full-scale war, but with casualties in the millions rather than the zero deaths that actually occurred.

Scenario Chain Analysis for Path 4A (Naval Confrontation):
A naval confrontation would have been the most unpredictable scenario. Soviet submarine B-59 actually came close to launching a nuclear torpedo when it lost communication and was being depth-charged by US destroyers. In the alternative scenario where orders were to proceed through the blockade, nuclear weapons use at sea would have been highly probable. This would have created unprecedented escalation pressures, as both sides would have faced the choice between nuclear retaliation or appearing weak after suffering nuclear attack.

Probability Assessment Framework:
Each alternative path can be assessed for its probability of leading to nuclear weapons use:
- Actual path taken: ~5% (crisis nearly escalated several times)
- Open deployment (1A): ~15% (longer crisis, more pressure points)  
- Surgical strikes (3A): ~85% (direct attack on Soviet forces)
- Naval confrontation (4A): ~90% (nuclear weapons already authorized for use)
- No diplomatic resolution (5B): ~95% (military action would have been inevitable)

Strategic Logic Analysis:
The counterfactual analysis reveals that Kennedy and Khrushchev made decisions that consistently chose options with lower escalation probability, even when those options appeared to offer fewer immediate advantages. This suggests that both leaders understood the escalation dynamics and were working to avoid nuclear conflict even while maintaining public positions of strength.

The critical insight from this analysis is that the Cuban Missile Crisis was resolved not through superior strategy by either side, but through mutual recognition that nuclear war would have been catastrophic for both superpowers regardless of who "won" the initial exchange.

Therefore, the counterfactual analysis demonstrates that while numerous alternative paths existed, most would have led to higher probabilities of nuclear weapons use. The actual resolution, while appearing to be the result of successful crisis management, was largely the product of both leaders consistently choosing de-escalation when faced with binary choices between face-saving and survival. This suggests that successful nuclear crisis management depends more on recognizing mutual vulnerability than on demonstrating superior resolve.
""",
        "params": {
            "max_tokens": 5500,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 20: Scientific Hypothesis Evaluation Chain (Evidence-Based Reasoning)",
        "prompt": """
CONTEXT:
A groundbreaking astronomical discovery has challenged existing theories about dark matter. Five competing hypotheses have emerged to explain the observed phenomena, each supported by different types of evidence. The scientific community must systematically evaluate these hypotheses to determine which best explains the complete set of observations.

Observational Data Summary:
The Vera Rubin Observatory detected 47 galaxy clusters exhibiting anomalous gravitational lensing patterns that cannot be explained by visible matter alone. These clusters, designated VR-2024-001 through VR-2024-047, show gravitational effects 8.3 times stronger than their visible matter would predict. However, unlike previous dark matter observations, these clusters exhibit three previously unknown characteristics: (1) gravitational effects that vary with temporal cycles averaging 14.7 Earth hours, (2) directional gravitational asymmetries that appear to point toward specific coordinates in space, and (3) electromagnetic signatures in the 847-MHz frequency range that correlate precisely with gravitational strength variations.

Supporting Observations from Multiple Instruments:
The James Webb Space Telescope confirmed the temporal variations in gravitational lensing strength across all 47 clusters, with precision measurements showing the 14.7-hour cycle is consistent to within 0.03 hours across clusters separated by billions of light-years. The Chandra X-ray Observatory detected corresponding variations in X-ray emissions from hot gas within the clusters, suggesting the gravitational changes are real rather than observational artifacts. The Event Horizon Telescope network provided ultra-high resolution imaging showing that the directional asymmetries point toward a region in the constellation Draco, approximately 23.7 degrees from the galactic north pole. Ground-based radio telescopes worldwide confirmed the 847-MHz emissions, with signal strength varying in perfect synchronization with gravitational effects.

Hypothesis 1: Modified Newtonian Dynamics (MOND) Extension
Dr. Sarah Chen's team proposes that the observations can be explained by a previously undetected modification to gravitational physics at galaxy cluster scales. Their hypothesis suggests that gravity becomes non-isotropic (directional) when acting on scales larger than 10 million light-years, creating preferential directions based on the overall structure of the universe. The temporal variations would result from galaxy clusters' rotation and orbital motion through this modified gravitational field. The electromagnetic signatures would be secondary effects caused by charged particles responding to the directional gravity variations.

Supporting Evidence: Mathematical models show that anisotropic gravity could reproduce the observed directional effects if the universe has a preferred reference frame, possibly related to dark energy flow patterns. Computer simulations using modified MOND equations can reproduce 89% of the observed gravitational lensing variations when temporal and directional factors are included. The 847-MHz frequency matches theoretical predictions for electromagnetic emissions from charged cosmic dust responding to oscillating gravitational fields.

Contradictory Evidence: MOND modifications have historically failed to explain phenomena at scales larger than individual galaxies. The required degree of anisotropy would have been detectable in previous gravitational wave experiments, yet none was observed. The 14.7-hour periodicity does not match any known cosmic rotation or orbital periods at the relevant scales.

Hypothesis 2: Exotic Dark Matter Interactions  
Dr. Michael Patel's research group argues for a new class of dark matter that interacts weakly with itself through previously unknown forces. This "interacting dark matter" would form complex structures analogous to atoms and molecules, but on cosmic scales. The temporal variations would result from dark matter orbital dynamics within these cosmic-scale structures. The directional effects would indicate that dark matter structures are aligned with large-scale cosmic filaments pointing toward the Draco region.

Supporting Evidence: Advanced particle physics models suggest that dark matter could have rich internal structure involving multiple particle types and force carriers. The 14.7-hour period matches theoretical predictions for orbital periods in cosmic-scale dark matter "molecules" bound by hypothetical dark sector forces. Cosmological simulations incorporating interacting dark matter produce large-scale structures with preferred orientations that could explain the Draco alignment.

Contradictory Evidence: Interacting dark matter would have left signatures in the cosmic microwave background that have not been observed. The required strength of dark sector forces would have affected dark matter's role in early universe structure formation, contradicting well-established cosmological models. Laboratory experiments have set strict limits on dark matter self-interaction strength that appear incompatible with the cosmic-scale structures this hypothesis requires.

Hypothesis 3: Primordial Black Hole Networks
Dr. Elena Rodriguez proposes that the observations result from networks of primordial black holes formed in the early universe. These black hole networks would have masses comparable to asteroid clusters but would be distributed throughout galaxy clusters in complex orbital configurations. The gravitational effects would result from these distributed black hole swarms, while temporal variations would arise from their orbital dynamics. The electromagnetic signatures would be produced by charged matter falling into the black holes.

Supporting Evidence: Primordial black holes with appropriate masses could have formed during specific epochs in the early universe when density fluctuations were optimal for black hole production. The 847-MHz emissions match theoretical predictions for radio signals from matter spiraling into small black holes. Gravitational microlensing surveys have detected unexplained events that could be consistent with primordial black hole populations.

Contradictory Evidence: The required density of primordial black holes would have been detectable through their effects on Big Bang nucleosynthesis, yet these effects are not observed. The orbital configurations needed to produce 14.7-hour periodicities would be gravitationally unstable over cosmic time scales. Hawking radiation from black holes of the required mass would have evaporated them billions of years ago.

Hypothesis 4: Higher-Dimensional Gravity Leakage
Dr. James Liu's theoretical framework suggests that gravity is "leaking" into our three-dimensional space from higher spatial dimensions, creating the appearance of dark matter. In this model, the temporal variations result from our galaxy cluster's motion through higher-dimensional space, while directional effects arise from the geometry of higher-dimensional gravitational fields. The electromagnetic signatures would be produced by higher-dimensional gravitational waves coupling to charged particles in our dimension.

Supporting Evidence: String theory and other fundamental physics theories predict extra spatial dimensions that could affect gravity at cosmological scales. The mathematical framework naturally explains why dark matter effects appear gravitational but don't interact electromagnetically. The Draco directional preference could indicate the orientation of higher-dimensional structures relative to our observable universe.

Contradictory Evidence: Higher-dimensional gravity effects should decrease with distance in ways that contradict observed dark matter distributions. Laboratory tests of gravity at small scales have found no evidence of extra-dimensional effects. The required coupling between higher dimensions and our three-dimensional space would have observable consequences for particle physics that have not been detected.

Hypothesis 5: Consciousness-Quantum Entanglement Field
Dr. Amanda Foster's controversial hypothesis proposes that consciousness throughout the universe creates a quantum field that manifests as apparent gravitational effects. According to this model, the 14.7-hour periodicity corresponds to circadian-like rhythms in cosmic consciousness, while directional effects point toward regions of highest consciousness density in the universe. The electromagnetic signatures would result from quantum field fluctuations associated with consciousness-matter interactions.

Supporting Evidence: Quantum mechanics suggests that consciousness plays a fundamental role in determining physical reality through observation and measurement. The 14.7-hour period is remarkably close to Earth's rotational period, suggesting a connection to biological rhythms. Some interpretations of quantum mechanics propose that consciousness extends beyond individual organisms to cosmic scales.

Contradictory Evidence: No known physical mechanism could link consciousness to gravitational effects at cosmological scales. The hypothesis makes no testable predictions that distinguish it from conventional dark matter models. The required consciousness density throughout the universe would have to be orders of magnitude greater than what could be produced by known biological processes.

SCIENTIFIC HYPOTHESIS EVALUATION:
To evaluate these competing hypotheses rigorously, I need to systematically weigh each piece of evidence and assess how well each theory explains the complete observational dataset. Let me start by categorizing the evidence types and establishing evaluation criteria for scientific validity.

Evidence Classification Framework:
The observational evidence falls into four categories: (1) gravitational effects (lensing strength, temporal variations, directional asymmetries), (2) electromagnetic signatures (847-MHz emissions, X-ray correlations), (3) spatial patterns (Draco alignment, cluster distribution), and (4) temporal characteristics (14.7-hour periodicity, cross-cluster synchronization). Each hypothesis must account for all four evidence categories to be considered scientifically viable.

Hypothesis 1 Evaluation - MOND Extension:
Strengths: Provides mathematical framework that reproduces 89% of lensing observations. Electromagnetic predictions match observed 847-MHz frequency. Builds on established MOND theory with specific extensions.
Weaknesses: Fails to explain lack of gravitational wave signatures. 14.7-hour periodicity not predicted by cosmic scales. Anisotropic gravity contradicts fundamental physics principles.
Scientific Viability: Moderate - makes testable predictions but faces fundamental physics challenges.

Hypothesis 2 Evaluation - Interacting Dark Matter:
Strengths: Consistent with particle physics extensions. Explains directional effects through cosmic filament alignment. Predicts correct orbital periods for cosmic structures.
Weaknesses: Contradicts cosmic microwave background observations. Self-interaction limits from laboratory experiments. Early universe structure formation problems.
Scientific Viability: High - despite challenges, makes specific testable predictions and builds on established physics.

Hypothesis 3 Evaluation - Primordial Black Holes:
Strengths: Electromagnetic signatures match theoretical predictions. Could explain microlensing anomalies. Uses well-understood black hole physics.
Weaknesses: Big Bang nucleosynthesis constraints. Hawking radiation evaporation timeline. Orbital stability problems over cosmic time.
Scientific Viability: Low - faces multiple observational constraints from different areas of physics.

Hypothesis 4 Evaluation - Higher-Dimensional Gravity:
Strengths: Naturally explains gravitational-only effects. Consistent with string theory framework. Could explain directional preferences.
Weaknesses: Distance-scaling contradicts observations. No laboratory evidence for extra dimensions. Particle physics predictions unfulfilled.
Scientific Viability: Low - lacks supporting evidence from multiple required areas.

Hypothesis 5 Evaluation - Consciousness Field:
Strengths: 14.7-hour period suggests biological connection. Quantum mechanics allows consciousness-matter interaction.
Weaknesses: No physical mechanism. No testable predictions. Requires unprecedented consciousness densities.
Scientific Viability: Negligible - fails basic scientific criteria for testable hypotheses.

Comparative Analysis:
When evaluated against the complete evidence set, Hypothesis 2 (Interacting Dark Matter) emerges as the most scientifically viable despite significant challenges. It makes specific, testable predictions and provides mechanisms for all four evidence categories. The cosmic microwave background contradiction could potentially be resolved through more sophisticated models of dark sector physics.

Hypothesis 1 (MOND Extension) ranks second, offering precise mathematical predictions but requiring fundamental revisions to gravitational physics. The other three hypotheses face insurmountable contradictions with well-established observations or physical principles.

Therefore, the systematic evaluation suggests that interacting dark matter provides the best current explanation for the anomalous observations, though significant theoretical development is needed to resolve contradictions with existing cosmological data. The scientific method requires that this hypothesis be subjected to rigorous experimental testing before acceptance, particularly through searches for dark sector particles and more precise cosmic microwave background analysis.
""",
        "params": {
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 21: Multi-Step Engineering Problem Decomposition (Structural Analysis)",
        "prompt": """
CONTEXT:
A critical infrastructure bridge has developed structural concerns and requires comprehensive analysis before determining whether emergency closure is necessary. The bridge carries 50,000 vehicles daily and closing it would create massive economic disruption. However, if the structure fails, casualties would be catastrophic. Engineers must systematically decompose this complex structural analysis into discrete, manageable steps.

Bridge Specifications and Current Status:
The Riverside Metropolitan Bridge is a 2,847-foot suspension bridge built in 1987 with a main span of 1,800 feet and two side spans of 523.5 feet each. The bridge deck is 90 feet wide, accommodating six lanes of traffic plus pedestrian walkways. The structure consists of two main towers (each 580 feet tall), main suspension cables (each containing 15,472 individual steel wires), 84 vertical suspender cables, and a steel truss deck system. Recent inspections have revealed five concerning issues: (1) visible corrosion on 23% of suspender cable connections, (2) hairline cracks in the north tower's concrete foundation extending to a depth of 18 inches, (3) unusual vibration patterns during high wind conditions exceeding design parameters by 15%, (4) deck surface deformation in three sections totaling 340 linear feet, and (5) bearing assembly displacement at the south anchorage measuring 3.2 inches from original position.

Load Analysis Requirements:
The bridge currently experiences daily traffic loads averaging 47,000 vehicles (85% passenger cars, 12% light trucks, 3% heavy commercial vehicles). Peak load conditions occur during morning and evening rush hours when traffic density reaches 95% of design capacity. Environmental loads include wind forces up to 85 mph (design maximum: 90 mph), seismic considerations for a zone 3 earthquake region, temperature variations from -15°F to 105°F causing thermal expansion and contraction, and ice loading during winter months adding up to 2.3 pounds per square foot additional load. Dynamic loads from traffic create vibration frequencies that have recently shown resonance patterns at 0.47 Hz and 1.23 Hz, which were not present in original design calculations.

Material Condition Assessment Data:
Laboratory analysis of cable samples shows 12% strength reduction from original specifications due to corrosion and fatigue. Concrete core samples from tower foundations reveal 18% reduction in compressive strength with chloride penetration depth averaging 2.1 inches. Steel deck components show surface rust on 34% of members but no significant cross-sectional loss. Bearing assemblies exhibit wear patterns consistent with 36 years of service but displacement suggests foundation settlement or thermal expansion issues. Paint system failure on 67% of steel surfaces has accelerated corrosion rates beyond normal maintenance projections.

Safety Factor Analysis:
Original design incorporated safety factors of 2.5 for live loads, 1.8 for wind loads, 2.0 for seismic loads, and 3.0 for dead loads. Current conditions suggest effective safety factors have been reduced to approximately 1.9 for live loads (due to cable degradation), 1.4 for wind loads (due to unexpected vibration resonance), 1.7 for seismic loads (due to foundation concerns), and 2.7 for dead loads (minimal change). Industry standards require minimum safety factors of 1.67 for live loads, 1.25 for wind loads, 1.5 for seismic loads, and 2.0 for dead loads under normal operating conditions.

STRUCTURAL ENGINEERING ANALYSIS:
To properly evaluate the bridge's safety and determine appropriate action, this complex structural problem must be systematically broken down into sequential analytical steps. Each step builds upon the previous analysis and leads to the next logical evaluation phase.

Step 1: Load Path Analysis and Critical Component Identification
Beginning with fundamental load path analysis, I must trace how forces flow through the structure and identify which components are critical to overall stability. The primary load path follows traffic loads from the deck surface through the steel truss system to the suspender cables, then to the main suspension cables, through the towers, and finally to the foundations and anchorages. Secondary load paths include wind forces acting on the deck and towers, seismic forces transmitted through the foundations, and thermal forces affecting the entire structure.

Critical components analysis reveals that the main suspension cables, tower foundations, and bearing assemblies represent single points of failure where damage could lead to catastrophic collapse. The suspender cables, while numerous, provide redundancy where individual cable failure would redistribute loads to adjacent cables. The deck truss system offers some load redistribution capability but depends entirely on the suspension system for primary support.

Step 2: Quantitative Degradation Assessment
Moving to quantitative analysis, I must calculate the precise impact of observed degradation on structural capacity. The 12% strength reduction in suspension cables directly reduces the bridge's live load capacity by approximately 10.8% (accounting for load redistribution). The 18% reduction in tower foundation strength creates a critical concern for overturning resistance under combined dead, live, and wind loads.

The unusual vibration patterns at 0.47 Hz and 1.23 Hz indicate potential aerodynamic instability. Comparing these frequencies to the bridge's natural frequencies (calculated as 0.52 Hz for first mode, 1.18 Hz for second mode) shows dangerous proximity that could lead to resonance amplification under specific wind conditions.

Step 3: Failure Mode Analysis and Probability Assessment
Systematically examining potential failure modes, I identify five critical scenarios: (1) progressive cable failure leading to load redistribution and eventual overload of remaining cables, (2) foundation failure under combined loading causing tower instability, (3) aerodynamic instability leading to destructive oscillations, (4) bearing failure causing deck displacement and load redistribution problems, and (5) fatigue failure of deck connections creating local collapse.

Probability analysis using current condition data suggests: Cable failure probability of 2.3% over the next 5 years under normal loading, foundation failure probability of 0.8% under design seismic loading, aerodynamic instability probability of 5.7% during severe wind events, bearing failure probability of 3.1% over the next 2 years, and deck connection failure probability of 1.2% over normal service life.

Step 4: Risk Assessment Matrix and Consequence Analysis  
Developing a comprehensive risk matrix, I must evaluate both probability and consequence for each failure mode. Cable failure would affect the entire bridge with potential for complete collapse, affecting 50,000 daily users plus surrounding areas. Foundation failure would similarly cause complete structural failure with catastrophic consequences. Aerodynamic instability could develop gradually, allowing for emergency closure, but would still represent extreme risk during sudden onset conditions.

Economic consequence analysis shows bridge closure would cost approximately $2.3 million daily in detour costs, delayed freight movement, and reduced economic activity. Complete failure would result in reconstruction costs of $400-600 million plus incalculable human casualties and long-term economic disruption.

Step 5: Load Capacity Recalculation with Current Conditions
Recalculating bridge capacity using degraded material properties and identified concerns: Revised live load capacity drops from 850 pounds per square foot to approximately 697 pounds per square foot (18% reduction). Wind load capacity decreases from 90 mph design maximum to approximately 76 mph due to aerodynamic concerns and foundation issues. Seismic capacity shows 23% reduction due to foundation degradation.

Current peak traffic loading reaches 643 pounds per square foot during rush hours, creating a capacity utilization of 92.2% under degraded conditions compared to the original design margin of 75.6%. This represents an unacceptable reduction in safety margin.

Step 6: Immediate Risk Mitigation Strategy Development
Based on the systematic analysis, immediate actions must be prioritized by risk level and implementation feasibility. Highest priority interventions include: implementing traffic restrictions to reduce live load to 65% of current levels (achievable through lane closures and weight restrictions), installing real-time structural monitoring systems to detect dangerous vibration patterns, and scheduling emergency inspections of all critical components on a weekly basis.

Medium-term interventions within 90 days should include: comprehensive cable replacement program starting with the most deteriorated sections, foundation strengthening through grouting and additional support systems, and installation of aerodynamic modifications to reduce wind-induced vibrations.

Step 7: Decision Framework and Recommendation Synthesis
Synthesizing all analytical steps into a coherent decision framework: The bridge currently operates with inadequate safety margins that fall below industry standards in multiple categories. The combination of degraded structural capacity and high utilization rates creates an unacceptable risk profile for continued unrestricted operation.

Recommendation: Implement immediate traffic restrictions reducing capacity to 65% while emergency repairs are undertaken. Complete closure is not yet necessary but should be prepared as a contingency measure if monitoring systems detect critical threshold violations.

Therefore, this systematic decomposition reveals that while the bridge can continue operating under restricted conditions with intensive monitoring, the degraded safety margins require immediate intervention to prevent a catastrophic failure that could result in significant loss of life and economic disruption. The step-by-step analysis provides a clear path forward that balances public safety with economic considerations while establishing measurable criteria for future decisions.
""",
        "params": {
            "max_tokens": 5000,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 22: Multi-Stage Medical Diagnosis Protocol (Systematic Differential Diagnosis)",
        "prompt": """
CONTEXT:
A 47-year-old patient presents to the emergency department with complex, multi-system symptoms that require systematic medical analysis. The case presents diagnostic challenges that must be approached through rigorous step-by-step differential diagnosis methodology to avoid misdiagnosis and ensure appropriate treatment.

Patient Presentation and Initial Assessment:
The patient, a 47-year-old software engineer, presents with a three-week history of progressive fatigue, intermittent fever (ranging from 100.2°F to 103.1°F), joint pain affecting multiple joints bilaterally, a distinctive rash on the face and upper chest, difficulty concentrating, and episodes of shortness of breath. The patient reports that symptoms began gradually following a camping trip in upstate New York but initially dismissed them as stress-related or viral illness. Family history includes autoimmune thyroid disease in mother and rheumatoid arthritis in maternal grandmother. The patient takes no regular medications, denies smoking, consumes 2-3 alcoholic beverages weekly, and maintains an active lifestyle including regular hiking and cycling.

Detailed Symptom Timeline and Characteristics:
Week 1: Mild fatigue and occasional headaches, attributed to work stress and lack of sleep. No fever documented, joint discomfort in hands and wrists dismissed as repetitive strain injury.

Week 2: Fatigue significantly worsened, first documented fever of 101.2°F, distinctive "butterfly" rash appeared across nose and cheeks, joint pain expanded to shoulders, knees, and ankles with morning stiffness lasting 2-3 hours. Patient experienced first episode of shortness of breath during routine exercise, which previously was well-tolerated.

Week 3: All symptoms intensified, peak fever of 103.1°F requiring ibuprofen, rash spread to upper chest and became photosensitive, joint pain now constant with significant impact on daily activities, cognitive symptoms emerged including difficulty concentrating and memory lapses, shortness of breath occurs with minimal exertion.

Physical Examination Findings:
Vital signs: Temperature 101.8°F, pulse 94 bpm, blood pressure 142/88 mmHg (elevated from patient's baseline of 125/75), respiratory rate 18 per minute, oxygen saturation 96% on room air. General appearance shows an alert but fatigued individual with obvious facial rash. Head and neck examination reveals no lymphadenopathy, thyroid normal to palpation, no jugular venous distension. Cardiovascular examination shows regular rate and rhythm, no murmurs or extra sounds, but mild peripheral edema in lower extremities. Pulmonary examination reveals clear lung fields bilaterally with no wheezes or crackles. Musculoskeletal examination demonstrates active synovitis in metacarpophalangeal joints, wrists, and knees bilaterally with warm, swollen joints and limited range of motion. Neurological examination shows intact cranial nerves, normal reflexes, but subtle cognitive slowing on mental status testing.

Laboratory Results and Diagnostic Testing:
Complete blood count: WBC 3,200/μL (low), hemoglobin 10.8 g/dL (low), hematocrit 32.1% (low), platelets 145,000/μL (low-normal), lymphocytes 15% (low). Basic metabolic panel: sodium 138 mEq/L, potassium 4.1 mEq/L, chloride 102 mEq/L, CO2 22 mEq/L, BUN 28 mg/dL (elevated), creatinine 1.4 mg/dL (elevated from baseline 0.9), glucose 98 mg/dL. Liver function tests: ALT 67 U/L (elevated), AST 58 U/L (elevated), alkaline phosphatase 89 U/L, total bilirubin 1.8 mg/dL (elevated). Inflammatory markers: ESR 78 mm/hr (markedly elevated), C-reactive protein 45 mg/L (markedly elevated). Urinalysis: protein 3+ (significant), RBCs 15-20/hpf (elevated), WBCs 5-10/hpf, RBC casts present (significant finding), specific gravity 1.025.

SYSTEMATIC MEDICAL DIAGNOSIS PROCESS:
This complex clinical presentation requires methodical analysis through established differential diagnosis protocols. Each diagnostic step must build systematically upon previous findings to reach an accurate conclusion while avoiding cognitive biases and premature closure.

Step 1: Symptom Pattern Recognition and System Involvement Assessment
Beginning with systematic symptom analysis, I identify involvement of multiple organ systems: constitutional symptoms (fever, fatigue), musculoskeletal system (polyarthritis), integumentary system (photosensitive rash), renal system (proteinuria, hematuria, RBC casts), hematologic system (cytopenias), and possible cardiovascular involvement (hypertension, edema). This multi-system involvement pattern suggests either systemic inflammatory disease, systemic infection, or malignancy.

The specific combination of butterfly rash, photosensitivity, polyarthritis, and renal involvement with RBC casts creates a pattern highly suggestive of systemic lupus erythematosus (SLE). However, systematic analysis requires consideration of other conditions that can mimic this presentation.

Step 2: Differential Diagnosis Generation Using Clinical Reasoning
Systematically generating differential diagnoses based on organ system involvement and symptom patterns:

Primary considerations include systemic lupus erythematosus (SLE) given the classic presentation, infectious endocarditis with systemic complications, systemic vasculitis such as polyarteritis nodosa or ANCA-associated vasculitis, adult-onset Still's disease, and drug-induced lupus-like syndrome.

Secondary considerations encompass tick-borne illnesses (given recent camping exposure), including Rocky Mountain spotted fever, ehrlichiosis, or Lyme disease with systemic complications, viral syndromes such as EBV, CMV, or parvovirus B19, hematologic malignancy with systemic manifestations, and mixed connective tissue disease.

Step 3: Evidence Weighting and Probability Assignment
Systematically evaluating each differential diagnosis against the clinical evidence:

SLE probability assessment: Meets 4 of 4 cardinal features (malar rash, arthritis, renal involvement, hematologic abnormalities). Laboratory findings strongly support with cytopenias, elevated inflammatory markers, and active urinary sediment. Estimated probability: 85%.

Infectious endocarditis probability: Fever and multi-system involvement fit, but lack of cardiac murmur, normal echocardiogram findings implied by physical exam, and specific rash pattern make this less likely. Estimated probability: 15%.

Systemic vasculitis probability: Could explain multi-system involvement and renal findings, but specific rash pattern and joint involvement pattern more consistent with SLE. Estimated probability: 10%.

Step 4: Targeted Diagnostic Testing Strategy
Based on probability analysis, systematic testing approach should prioritize confirming or excluding SLE while ruling out serious alternative diagnoses:

First-tier testing: ANA (antinuclear antibody) with pattern and titer, anti-dsDNA antibodies, anti-Smith antibodies, complement levels (C3, C4), blood cultures to exclude endocarditis, and echocardiogram to assess cardiac involvement.

Second-tier testing if first-tier supports SLE: Anti-SSA/Ro, anti-SSB/La, anti-RNP antibodies, lupus anticoagulant and anticardiolipin antibodies, and renal biopsy to assess degree of lupus nephritis.

Exclusion testing for alternative diagnoses: Tick-borne illness serologies given exposure history, hepatitis panel given elevated liver enzymes, and flow cytometry if hematologic malignancy suspected.

Step 5: Risk Stratification and Urgency Assessment
Systematically assessing disease severity and treatment urgency:

High-risk features present include active nephritis with RBC casts and proteinuria, cytopenias with potential bleeding or infection risk, elevated blood pressure suggesting renal involvement, and multi-system inflammatory process with high inflammatory markers.

Immediate interventions required: nephrology consultation for renal biopsy consideration, blood pressure management, monitoring for bleeding complications due to thrombocytopenia, and infection precautions due to leukopenia.

The presence of active nephritis elevates this from a routine SLE evaluation to an urgent case requiring immediate treatment initiation to prevent permanent renal damage.

Step 6: Treatment Planning and Monitoring Strategy
Systematically developing treatment approach based on most likely diagnosis:

If SLE confirmed with active nephritis: Immediate immunosuppressive therapy with high-dose corticosteroids, consideration of steroid-sparing agents such as mycophenolate mofetil or cyclophosphamide, ACE inhibitor for renal protection and blood pressure control, and hydroxychloroquine for long-term disease management.

Monitoring parameters: Daily assessment of renal function, blood pressure, and blood counts during acute treatment phase, weekly laboratory monitoring during immunosuppressive therapy initiation, and long-term monitoring for treatment complications and disease activity.

Step 7: Comprehensive Care Coordination and Patient Education
Systematically addressing broader care needs: Rheumatology consultation for long-term management, ophthalmology screening before hydroxychloroquine initiation, patient education regarding disease nature, treatment expectations, and lifestyle modifications including sun protection and infection precautions.

Therefore, systematic analysis of this complex presentation using structured diagnostic methodology points to systemic lupus erythematosus with active nephritis as the most likely diagnosis, requiring immediate treatment initiation to prevent irreversible organ damage while confirming the diagnosis through targeted laboratory testing and specialist consultation.
""",
        "params": {
            "max_tokens": 5500,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 23: Multi-Phase Investigation Protocol (Corporate Fraud Analysis)",
        "prompt": """
CONTEXT:
A multinational corporation with annual revenues of $8.7 billion is suspected of systematic financial fraud affecting multiple divisions and geographic regions. The investigation must be conducted through rigorous step-by-step forensic analysis to uncover the full scope of fraudulent activities while ensuring legal admissibility of evidence and protecting legitimate business operations.

Initial Fraud Indicators and Whistleblower Reports:
The investigation began when a senior financial analyst in the European division contacted internal audit with concerns about revenue recognition practices in three major client contracts totaling €47 million. Subsequent anonymous tips through the corporate ethics hotline revealed: (1) suspected manipulation of quarterly earnings through premature revenue recognition, (2) possible related-party transactions disguised as arm's-length deals, (3) inventory valuation irregularities across manufacturing divisions, (4) expense timing manipulation to smooth earnings, and (5) potential bribery of government officials in emerging market operations. The whistleblower provided specific account numbers, dates, and personnel names, along with copied documents showing email communications discussing "revenue smoothing techniques" and "creative accounting approaches."

Financial Statement Anomalies and Red Flag Analysis:
Preliminary analysis of the company's financial statements over the past four years reveals several concerning patterns: revenue growth rates that significantly exceed industry benchmarks (averaging 23% annually vs. industry average of 8%), gross margins that remain suspiciously stable despite volatile raw material costs, accounts receivable growing faster than sales (indicating possible fictitious sales), inventory turnover rates declining while management reports improved efficiency, and unusual spikes in revenue during final weeks of each quarter. Days Sales Outstanding (DSO) increased from 45 days to 78 days over three years without corresponding changes in customer payment terms. The company's relationship with its external auditor shows concerning patterns of partner rotation and disagreements over accounting treatments that were subsequently overruled by management.

Organizational Structure and Key Personnel:
The corporation operates through seven major divisions across 23 countries, with complex reporting relationships that may facilitate fraud concealment. Key personnel under investigation include: the Chief Financial Officer who joined three years ago from a company previously sanctioned for accounting irregularities, the Vice President of Sales who receives unusually large bonuses based on quarterly performance, the Controller of International Operations who has authority over revenue recognition for overseas contracts, the Director of Procurement whose department handles significant vendor relationships, and the Regional Manager for Emerging Markets who oversees operations in countries with high corruption risk indices. The company's board of directors includes several members with limited financial expertise, and the audit committee meets only quarterly despite the company's size and complexity.

Transaction Analysis and Pattern Identification:
Detailed transaction analysis reveals systematic patterns suggesting coordinated fraud: quarter-end journal entries that are reversed in the subsequent period, consistent overriding of system-generated accounting entries by senior management, transactions with vendors that lack normal supporting documentation, revenue recognition timing that accelerates as quarters progress, and expense accruals that are consistently understated and subsequently adjusted. Bank account analysis shows unusual cash movements including transfers to accounts in jurisdictions known for banking secrecy, payments to consulting companies with no clear business purpose, and cash advances to employees that are never reconciled or repaid.

FORENSIC INVESTIGATION METHODOLOGY:
This complex corporate fraud investigation requires systematic analysis through established forensic accounting protocols. Each investigative phase must build methodically upon previous findings while maintaining legal standards and operational security.

Phase 1: Evidence Preservation and Investigation Planning
Beginning with immediate evidence preservation, I must secure all relevant documentation, electronic records, and communication systems before evidence can be altered or destroyed. This includes placing litigation holds on email systems, securing accounting records and supporting documentation, imaging computer hard drives of key personnel, obtaining bank records for all corporate accounts, and securing physical documents in fireproof storage with chain-of-custody documentation.

Investigation planning requires establishing the investigative team with appropriate expertise: certified fraud examiners for financial analysis, computer forensics specialists for electronic evidence, legal counsel for privilege and admissibility issues, external auditors for independent verification, and industry experts for transaction validation. The investigation scope must be defined to include all potentially affected business units, time periods extending back five years to capture pattern development, and geographic regions where irregularities have been identified.

Phase 2: Financial Statement Analysis and Analytical Procedures
Systematically analyzing financial statements using forensic accounting techniques to identify specific areas of manipulation: Benford's Law analysis of journal entries to detect artificial number patterns, trend analysis comparing the company to industry benchmarks and historical performance, ratio analysis focusing on liquidity, profitability, and efficiency metrics that may indicate manipulation, and regression analysis to identify unusual relationships between related accounts.

Specific analytical procedures include calculating the probability of achieving exactly forecasted earnings (statistically improbable without manipulation), analyzing the timing of revenue recognition relative to cash receipts, examining the relationship between reported sales and underlying business drivers, and identifying accounts with unusual activity patterns or balances that deviate from expected norms.

Phase 3: Detailed Transaction Testing and Sampling
Systematically testing specific transactions using statistical sampling and targeted selection methods: Revenue transactions selected based on size, timing, customer risk factors, and unusual characteristics, expense transactions focusing on areas with high manipulation risk such as accruals, reserves, and discretionary expenses, related-party transactions requiring enhanced scrutiny for arm's-length nature and business purpose, and cash transactions examining the business rationale and authorization for unusual payments or transfers.

Each selected transaction undergoes comprehensive testing including verification of underlying business rationale, examination of supporting documentation for authenticity and completeness, confirmation with external parties where appropriate, and analysis of management authorization and approval processes.

Phase 4: Interview Protocol and Witness Statement Development
Systematically conducting interviews using established forensic interviewing techniques: Beginning with lower-risk individuals to gather background information and identify key issues, progressing to witnesses with direct knowledge of suspected fraudulent activities, and concluding with interviews of subjects suspected of wrongdoing.

Interview strategy includes preparing detailed question protocols based on documentary evidence, conducting interviews in controlled environments with appropriate legal representation, documenting all statements with written summaries or recordings where permitted, and obtaining signed witness statements where appropriate. Each interview builds upon previous findings and helps establish the timeline, methodology, and scope of fraudulent activities.

Phase 5: Electronic Evidence Analysis and Data Mining
Systematically analyzing electronic evidence using forensic technology tools: Email analysis focusing on communications between key personnel regarding revenue recognition, accounting treatments, and business transactions, database analysis examining patterns of data manipulation or unauthorized changes to financial records, and metadata analysis of electronic documents to identify creation dates, modification history, and authorship.

Data mining techniques include keyword searches for terms associated with fraud or manipulation, timeline analysis to correlate communications with financial statement preparation periods, and network analysis to identify communication patterns between individuals involved in suspicious activities.

Phase 6: Expert Analysis and Independent Verification
Systematically obtaining independent expert analysis to validate findings: Industry experts to assess whether business transactions reflect normal industry practices, valuation experts to determine appropriate accounting treatment for complex transactions, and computer forensics experts to authenticate electronic evidence and establish digital timelines.

Independent verification includes confirming significant transactions with external parties, obtaining bank confirmations for unusual account activity, and verifying the existence and business purpose of entities involved in suspicious transactions.

Phase 7: Report Preparation and Recommendation Development
Systematically preparing comprehensive investigation report documenting findings, conclusions, and recommendations: Executive summary highlighting key findings and financial impact, detailed analysis section presenting evidence and conclusions for each area of investigation, supporting documentation including schedules, calculations, and exhibits, and recommendations for remedial action, internal control improvements, and potential legal action.

The report must meet legal standards for potential litigation use while clearly communicating complex financial concepts to various stakeholders including board members, regulatory authorities, and potential law enforcement agencies.

Therefore, this systematic forensic investigation methodology provides a comprehensive framework for uncovering the full scope of suspected corporate fraud while ensuring evidence integrity and legal admissibility, ultimately supporting informed decision-making regarding remedial actions and potential prosecution of responsible parties.
""",
        "params": {
            "max_tokens": 5800,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 24: Multi-Stage Environmental Impact Assessment (Systematic Ecological Analysis)",
        "prompt": """
CONTEXT:
A proposed large-scale renewable energy project involving construction of a 500-turbine wind farm across 47,000 acres of mixed-use land requires comprehensive environmental impact assessment. The project spans multiple ecosystems, affects several endangered species, and involves complex interactions with existing land uses, requiring systematic step-by-step analysis to evaluate environmental consequences and mitigation strategies.

Project Specifications and Geographic Context:
The proposed Midwest Regional Wind Energy Project would be constructed across portions of five counties in the central Great Plains region, encompassing 47,000 acres of mixed agricultural land, native prairie remnants, wetland complexes, and forested creek bottoms. The project includes 500 wind turbines (each 450 feet tall), 250 miles of internal access roads, 180 miles of electrical transmission lines, 23 electrical substations, a central maintenance facility covering 50 acres, and temporary construction areas totaling 1,200 acres. The turbines would be arranged in clusters with minimum spacing of 1,000 feet between units, following topographic ridgelines and areas of optimal wind resource. Construction is planned over a three-year period, with operations continuing for 30 years followed by decommissioning and site restoration.

Baseline Ecological Assessment and Sensitive Species:
The project area encompasses several distinct ecological zones: 32,000 acres of agricultural land primarily in corn and soybean rotation, 8,500 acres of native tallgrass prairie including high-quality remnants, 4,200 acres of seasonal wetlands providing waterfowl habitat, 2,300 acres of riparian forest corridors along three creek systems, and scattered woodlots totaling approximately 800 acres. Wildlife surveys have documented 247 bird species, including 23 species of conservation concern, 45 mammal species including white-tailed deer, coyotes, and several bat species, and 18 amphibian species dependent on wetland habitats.

Endangered and threatened species confirmed in the project area include the greater prairie-chicken (state endangered) with two active lek sites within the project boundary, the northern long-eared bat (federally threatened) using forested areas for roosting, the poweshiek skipperling butterfly (federally endangered) in native prairie remnants, the least bittern (state threatened) nesting in emergent wetland vegetation, and the ornate box turtle (species of special concern) using prairie and edge habitats.

Hydrology and Water Resource Analysis:
The project area contains complex hydrological systems including portions of three major watershed basins affecting regional water quality and flood control. Surface water features include 847 seasonal wetlands classified as jurisdictional waters under the Clean Water Act, 34 miles of intermittent and perennial streams providing fish habitat and water quality functions, two natural lakes totaling 145 acres supporting waterfowl and fish populations, and artificial drainage systems including 127 miles of agricultural tiles and constructed drainage ditches. Groundwater resources include the regional Ogallala Aquifer providing irrigation water for agricultural operations, shallow alluvial aquifers along creek corridors, and perched water tables in some areas that support wetland hydrology.

Soil Resources and Agricultural Productivity:
Soil surveys indicate the project area contains predominantly highly productive agricultural soils classified as prime farmland, including Mollisols with high organic matter content and excellent structure for crop production. Specific soil types include 18,500 acres of Class I soils (highest agricultural capability), 21,200 acres of Class II soils with minor limitations, 5,800 acres of Class III soils with moderate limitations, and 1,500 acres of Class IV-VIII soils with severe limitations for agriculture. Many areas show evidence of historical soil erosion from intensive agriculture, with some restoration needed to maintain long-term productivity.

SYSTEMATIC ENVIRONMENTAL IMPACT ANALYSIS:
This complex environmental assessment requires methodical analysis through established ecological impact assessment protocols. Each analytical phase must systematically build upon baseline data to predict, quantify, and mitigate potential environmental consequences.

Phase 1: Impact Identification and Scoping Analysis
Beginning with systematic identification of all potential environmental impacts across construction, operation, and decommissioning phases. Construction impacts include habitat disturbance from turbine foundations (500 sites × 0.5 acres = 250 acres permanently affected), temporary disturbance from access roads and construction staging areas (approximately 2,800 acres temporarily affected), soil erosion and sedimentation during earthwork activities, noise and visual disturbance affecting wildlife behavior, and potential impacts to wetlands and streams from infrastructure placement.

Operational impacts include bird and bat mortality from turbine collisions (estimated 2-5 birds per turbine per year, 5-15 bats per turbine per year based on regional studies), habitat fragmentation effects from roads and turbine spacing, ongoing noise impacts on sensitive wildlife species, landscape visual impacts affecting recreational and aesthetic values, and electromagnetic interference with wildlife navigation systems.

Decommissioning impacts include temporary disturbance during removal activities, potential soil contamination from equipment and materials, and long-term land use changes if complete restoration is not achieved.

Phase 2: Species-Specific Impact Assessment and Population Modeling
Systematically analyzing impacts on each species of conservation concern using population viability analysis and habitat suitability modeling. Greater prairie-chicken impact assessment indicates the two active leks are located within 2,000 meters of proposed turbine locations, which research suggests could cause lek abandonment due to behavioral avoidance of tall structures. Population modeling suggests loss of these two leks could reduce regional population by 15-20% given limited alternative habitat.

Northern long-eared bat assessment involves analyzing roosting habitat loss (estimated 23 acres of forested areas directly affected) and mortality risk from turbine collisions during migration periods. Species-specific modeling indicates potential for 50-150 individual bat mortalities annually across the project, representing a significant impact given the species' declining population status.

Poweshiek skipperling butterfly assessment focuses on prairie habitat quality and connectivity. The project would directly impact 340 acres of native prairie habitat, including two high-quality sites supporting confirmed populations. Fragmentation analysis indicates remaining habitat patches may fall below minimum viable population thresholds.

Phase 3: Ecosystem Service Valuation and Cumulative Impact Analysis
Systematically quantifying ecosystem services provided by existing habitats and calculating economic values of potential losses. Prairie ecosystem services include carbon sequestration (estimated 2.3 tons CO2 per acre per year), soil retention and water quality protection, pollinator habitat supporting agricultural productivity, and flood control through water infiltration and retention.

Wetland ecosystem services include water purification (estimated value $8,400 per acre annually), flood control storage capacity (average 2.8 acre-feet per wetland), wildlife habitat supporting hunting and ecotourism (estimated $450 per acre annually), and groundwater recharge functions. Total ecosystem service value for areas directly impacted by the project is estimated at $12.7 million annually.

Cumulative impact analysis examines this project in context with existing and planned development in the region, including two existing wind farms within 15 miles, proposed transmission line upgrades, agricultural intensification, and urban development pressures.

Phase 4: Mitigation Hierarchy Application and Compensation Planning
Systematically applying the mitigation hierarchy of avoid, minimize, restore, and offset to develop comprehensive mitigation strategy. Avoidance measures include relocating 47 turbines away from high-quality prairie remnants and sensitive wildlife areas, establishing 500-meter buffer zones around confirmed greater prairie-chicken leks, routing access roads to avoid wetland complexes, and timing construction activities to avoid critical breeding seasons.

Minimization measures include using tubular tower designs to reduce perching opportunities for raptors, implementing low-impact construction techniques in sensitive areas, installing wildlife-friendly lighting systems, and developing adaptive management protocols for operational modifications based on monitoring results.

Restoration and offset measures include restoring 850 acres of degraded agricultural land to native prairie habitat, enhancing 1,200 acres of existing prairie through invasive species control and prescribed burning, creating 45 acres of wetland habitat to compensate for unavoidable impacts, and contributing $2.3 million to regional conservation programs targeting species of concern.

Phase 5: Monitoring and Adaptive Management Protocol Development
Systematically designing monitoring programs to track environmental impacts and evaluate mitigation effectiveness. Pre-construction monitoring establishes comprehensive baseline data for all identified impact categories, including wildlife population surveys, habitat condition assessments, water quality monitoring, and noise level measurements.

Construction monitoring includes daily environmental compliance inspections, wildlife mortality documentation, erosion and sediment control effectiveness, and restoration progress tracking. Post-construction monitoring continues for the project's 30-year operational life, including annual bird and bat mortality surveys, greater prairie-chicken lek activity monitoring, vegetation establishment success on restored areas, and adaptive management trigger evaluations.

Phase 6: Regulatory Compliance and Permit Acquisition Strategy
Systematically addressing all applicable environmental regulations and permit requirements. Federal requirements include Clean Water Act Section 404 permits for wetland impacts, Endangered Species Act consultation for all listed species, National Environmental Policy Act documentation, and Federal Aviation Administration clearances for turbine heights.

State and local requirements include state environmental review processes, wildlife take permits for unavoidable mortality, air quality permits for construction activities, and local zoning and land use approvals.

Phase 7: Stakeholder Engagement and Public Participation Integration
Systematically engaging affected stakeholders throughout the assessment process. Stakeholder identification includes adjacent landowners, agricultural operators, conservation organizations, tribal governments, recreational users, and local communities. Engagement methods include public scoping meetings, technical advisory committees, stakeholder interviews, and public comment periods on draft assessments.

Stakeholder input integration involves incorporating local ecological knowledge, addressing community concerns about visual and noise impacts, coordinating with agricultural operations to minimize disruption, and developing community benefit programs that share project revenues for local conservation initiatives.

Therefore, this systematic environmental impact assessment provides a comprehensive framework for evaluating complex ecological consequences while developing effective mitigation strategies that balance renewable energy development goals with environmental protection requirements, ensuring long-term sustainability of both human and natural systems.
""",
        "params": {
            "max_tokens": 6200,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 25: Multi-Layer Cybersecurity Incident Response (Systematic Threat Analysis)",
        "prompt": """
CONTEXT:
A Fortune 500 financial services company has detected a sophisticated cyber attack that has compromised multiple systems across their global network. The incident requires immediate, systematic response following established cybersecurity frameworks to contain the threat, assess damage, preserve evidence, and restore operations while maintaining regulatory compliance and customer trust.

Initial Attack Detection and Alert Analysis:
The company's Security Operations Center (SOC) received the first alert at 14:23 UTC when an automated intrusion detection system flagged unusual network traffic patterns originating from the Eastern European region. Subsequent analysis revealed: (1) successful penetration of the corporate VPN through compromised employee credentials, (2) lateral movement across internal networks affecting customer data systems, (3) installation of advanced persistent threat (APT) malware on 47 workstations and 12 servers, (4) unauthorized access to customer account databases containing personally identifiable information (PII) for approximately 890,000 customers, and (5) evidence of data exfiltration attempts through encrypted channels to external command and control servers.

The attack vector analysis indicates initial compromise occurred through a spear-phishing email sent to 23 employees in the treasury operations department, with 3 employees clicking malicious links that installed credential harvesting malware. The attackers then used harvested domain administrator credentials to escalate privileges and move laterally through the network, targeting high-value systems including core banking applications, customer relationship management databases, and regulatory reporting systems.

Network Architecture and Affected Systems:
The company's IT infrastructure consists of a hybrid cloud environment with on-premises data centers in New York, London, and Singapore, plus cloud-based services hosted on Amazon Web Services and Microsoft Azure. Affected systems include: the primary customer database server containing 2.3 million customer records, three application servers hosting online banking services used by 450,000 active customers daily, the enterprise resource planning system containing financial and operational data, email servers processing 50,000 messages daily, and desktop workstations used by treasury operations, customer service, and IT administration staff.

Network segmentation analysis reveals the attackers successfully bypassed firewall controls between the corporate network and the regulated customer data environment, suggesting either insider assistance or previously unknown network vulnerabilities. The company's network monitoring systems show evidence of reconnaissance activities dating back approximately 6 weeks, indicating this was a planned, targeted attack rather than opportunistic malware.

Regulatory and Compliance Implications:
As a financial services organization, the company operates under multiple regulatory frameworks including the Gramm-Leach-Bliley Act, Payment Card Industry Data Security Standard (PCI DSS), Sarbanes-Oxley Act, and international regulations such as the European Union's General Data Protection Regulation (GDPR). The incident potentially affects customers across 23 countries, each with different breach notification requirements and timelines.

Regulatory notification requirements include informing the Federal Financial Institutions Examination Council within 36 hours, notifying affected state attorneys general within 72 hours for residents in their jurisdictions, providing customer notification within varying timeframes (24-72 hours depending on jurisdiction), and reporting to international regulators including the UK's Financial Conduct Authority and the European Central Bank for affected European customers.

SYSTEMATIC INCIDENT RESPONSE METHODOLOGY:
This critical cybersecurity incident requires immediate, coordinated response following established incident response frameworks. Each response phase must be executed systematically to contain damage while preserving evidence and maintaining business continuity.

Phase 1: Immediate Containment and Isolation
Beginning with immediate threat containment to prevent further damage and data loss. Network isolation procedures include disconnecting affected systems from external internet access while maintaining internal connectivity for forensic analysis, implementing emergency firewall rules to block communication with identified command and control servers, quarantining compromised workstations using endpoint detection and response tools, and establishing secure communication channels for incident response team coordination.

System isolation priorities focus first on customer-facing systems to prevent ongoing data access, followed by critical infrastructure systems essential for business operations, and finally administrative systems that may contain additional credentials or sensitive information. Each isolation action must be logged and documented for later forensic analysis and regulatory reporting.

Credential management includes immediately disabling all potentially compromised user accounts, forcing password resets for all administrative accounts, implementing enhanced multi-factor authentication requirements, and reviewing privileged access lists to ensure appropriate authorization levels.

Phase 2: Comprehensive Damage Assessment and Evidence Preservation
Systematically evaluating the full scope of compromise while preserving digital evidence for forensic analysis and potential legal proceedings. Evidence preservation includes creating forensic images of all affected systems before any remediation activities, maintaining detailed chain of custody documentation for all collected evidence, preserving network traffic logs and security system alerts, and coordinating with legal counsel to ensure evidence preservation meets potential litigation requirements.

Damage assessment methodology involves analyzing log files to establish timelines of attacker activities, identifying all systems accessed by attackers using network flow analysis and endpoint forensics, determining what data was accessed, modified, or exfiltrated through database transaction logs and file access records, and assessing integrity of critical business data and financial records.

Customer data impact assessment includes identifying specific customer records accessed by attackers, categorizing types of personal information potentially compromised (names, addresses, account numbers, Social Security numbers, financial data), analyzing transaction histories for evidence of unauthorized account access or fraudulent activities, and preparing preliminary customer notification materials.

Phase 3: Threat Intelligence Analysis and Attribution
Systematically analyzing attack methodologies, tools, and techniques to identify threat actors and predict future actions. Malware analysis includes reverse engineering identified malware samples to understand capabilities and persistence mechanisms, analyzing command and control communications to identify external infrastructure and potential future targets, and comparing attack signatures against known threat actor profiles and tactics, techniques, and procedures (TTPs).

Attribution analysis involves correlating attack indicators with known advanced persistent threat (APT) groups, analyzing linguistic patterns and metadata in malware and communication channels, investigating geolocation data for attack sources and infrastructure, and coordinating with government agencies and industry partners for threat intelligence sharing.

The analysis suggests attribution to APT29 (Cozy Bear), a sophisticated threat actor associated with Russian intelligence services, based on malware signatures, infrastructure patterns, and targeting preferences consistent with previous campaigns against financial institutions.

Phase 4: Stakeholder Communication and Regulatory Notification
Systematically managing communications to internal stakeholders, customers, regulators, and media while maintaining operational security and legal requirements. Internal communication includes briefing executive leadership and board of directors on incident scope and response actions, coordinating with legal counsel and compliance teams on regulatory obligations, engaging public relations teams for potential media response, and updating employees on security measures and operational changes.

Regulatory notification procedures involve preparing detailed incident reports for financial regulators including timelines, affected systems, and customer impact assessments, coordinating with law enforcement agencies for potential criminal investigation support, filing required breach notifications with state and international authorities, and preparing customer notification letters meeting legal requirements for content and timing.

Customer communication strategy includes preparing clear, factual notifications explaining what happened, what information was involved, what the company is doing in response, and what customers should do to protect themselves, establishing dedicated customer service lines to handle inquiries and concerns, and coordinating with credit monitoring services to provide identity protection services for affected customers.

Phase 5: System Recovery and Business Continuity Implementation
Systematically restoring operations while ensuring compromised systems are fully cleaned and secured. Recovery planning includes prioritizing system restoration based on business criticality and customer impact, implementing enhanced security controls before bringing systems back online, conducting thorough vulnerability assessments and penetration testing of recovered systems, and developing rollback procedures in case additional compromise is discovered.

Business continuity measures involve activating backup systems and disaster recovery sites for critical operations, implementing manual processes for essential business functions while systems are being restored, coordinating with vendors and service providers to maintain essential services, and updating business continuity plans based on lessons learned from the incident.

Security enhancement includes deploying additional monitoring and detection capabilities to identify similar future attacks, implementing network segmentation improvements to limit lateral movement, enhancing email security controls to prevent spear-phishing attacks, and providing additional cybersecurity training for all employees.

Phase 6: Long-term Monitoring and Threat Hunting
Systematically monitoring for signs of continued compromise or follow-up attacks while implementing proactive threat hunting activities. Enhanced monitoring includes deploying advanced behavioral analytics to detect subtle signs of ongoing attacker presence, implementing continuous network monitoring with enhanced logging and alerting capabilities, conducting regular forensic analysis of critical systems, and establishing threat intelligence feeds to identify emerging threats targeting the financial services sector.

Proactive threat hunting activities include analyzing network traffic patterns for indicators of advanced persistent threats, conducting regular vulnerability assessments and penetration testing, reviewing access logs and user behavior patterns for signs of insider threats, and participating in industry threat intelligence sharing programs.

Phase 7: Lessons Learned Integration and Security Program Enhancement
Systematically analyzing the incident to identify security program improvements and prevent similar future attacks. Post-incident analysis includes conducting comprehensive root cause analysis to identify all contributing factors, evaluating effectiveness of existing security controls and incident response procedures, identifying gaps in detection and response capabilities, and developing recommendations for security program enhancements.

Security program improvements include updating incident response plans based on lessons learned, enhancing employee security awareness training programs, implementing advanced threat detection technologies, strengthening third-party vendor security requirements, and establishing continuous security monitoring and assessment capabilities.

Therefore, this systematic cybersecurity incident response methodology provides a comprehensive framework for managing complex security incidents while meeting regulatory requirements, preserving customer trust, and strengthening organizational security capabilities to prevent future attacks.
""",
        "params": {
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 26: Self-Verification Legal Reasoning (Double-Check Analysis)",
        "prompt": """
CONTEXT:
A complex constitutional law case requires analysis with mandatory self-verification at each reasoning step. The case involves multiple constitutional principles that may conflict, requiring careful analysis followed by systematic verification of the reasoning process.

Constitutional Law Case: Digital Privacy vs. National Security
The case involves the government's use of artificial intelligence systems to analyze private communications without warrants, justified under national security provisions. Three constitutional principles are in tension: Fourth Amendment privacy protections, First Amendment free speech implications, and congressional authority to regulate interstate commerce and national defense.

Legal Precedent Framework:
Key precedents include Katz v. United States (1967) establishing reasonable expectation of privacy, Smith v. Maryland (1979) limiting privacy expectations for metadata, Carpenter v. United States (2018) extending privacy protections to digital location data, and various national security cases including United States v. United States District Court (1972) and Clapper v. Amnesty International (2013).

The Current Case Facts:
Government agencies deployed AI systems to analyze communication patterns, content themes, and social network connections of 2.3 million citizens without individual warrants, using a blanket national security authorization. The analysis identified potential threats but also flagged political dissidents, journalists, and activists based on communication content and association patterns.

CONSTITUTIONAL ANALYSIS WITH VERIFICATION LOOPS:
This complex constitutional analysis requires systematic reasoning followed by mandatory verification of each analytical step to ensure logical consistency and legal accuracy.

Initial Analysis Phase:
The Fourth Amendment analysis suggests that bulk AI surveillance without individual suspicion violates reasonable privacy expectations established in Carpenter. The government's use of content analysis, not just metadata, extends beyond the Smith v. Maryland exception and requires warrant protection under traditional Fourth Amendment jurisprudence.

Verification Check #1: Let me verify this initial Fourth Amendment reasoning by examining potential counterarguments and ensuring I haven't overlooked relevant precedent or misapplied existing law.

Upon verification, my initial analysis appears sound but incomplete. I need to consider the "special needs" doctrine from cases like Vernonia School District v. Acton (1995), which allows warrantless searches in certain contexts. However, the Supreme Court has been reluctant to extend this doctrine to general law enforcement, particularly for bulk surveillance without individualized suspicion.

The verification also reveals I should address the third-party doctrine more thoroughly. While Smith v. Maryland allowed metadata collection, Carpenter specifically rejected extending this to comprehensive digital tracking. The AI analysis here goes beyond metadata to include content analysis and behavioral pattern recognition, which Carpenter suggests requires warrant protection.

Verified Fourth Amendment Conclusion: The bulk AI surveillance program likely violates the Fourth Amendment because it involves comprehensive analysis of private communications content and patterns without individualized suspicion, extending well beyond the limited third-party doctrine exceptions recognized in Smith v. Maryland and explicitly rejected in Carpenter v. United States.

First Amendment Analysis Phase:
The First Amendment implications center on the chilling effect doctrine from cases like Laird v. Tatum (1972) and more recently, Clapper v. Amnesty International (2013). Government surveillance of communication content and association patterns could deter protected speech and association, particularly affecting journalists, activists, and political dissidents.

Verification Check #2: Let me verify this First Amendment analysis by examining whether I've properly applied the chilling effect doctrine and considered all relevant precedent regarding surveillance and free expression.

Upon verification, I realize my analysis needs refinement. Clapper actually raised the standing bar for chilling effect claims by requiring more concrete evidence of harm rather than speculative concerns. However, this case differs from Clapper because the surveillance is confirmed and comprehensive, not speculative. The AI analysis specifically flagged political dissidents and journalists, demonstrating actual rather than hypothetical targeting of protected speech.

I should also consider the associational privacy doctrine from NAACP v. Alabama (1958) and its modern applications. The AI system's analysis of social network connections directly implicates associational privacy rights, particularly when political and advocacy groups are systematically analyzed.

Verified First Amendment Conclusion: The surveillance program likely violates First Amendment associational privacy rights by systematically analyzing communication patterns and social connections of political activists and journalists. Unlike the speculative harm in Clapper, this case involves confirmed surveillance that has actually identified and flagged individuals based on protected speech and association activities.

National Security Authority Analysis:
Congressional authority for national security surveillance stems from various sources including the Commerce Clause, Necessary and Proper Clause, and inherent executive powers in foreign affairs. Key precedent includes the Foreign Intelligence Surveillance Act framework and cases like In re Sealed Case (2002) and United States v. United States District Court (1972).

Verification Check #3: Let me verify my understanding of national security authority by ensuring I've properly balanced congressional authorization against constitutional limitations and considered the scope of executive power in this context.

Upon verification, I find my analysis must address the Authorization for Use of Military Force interpretations and their limitations as established in Hamdi v. Rumsfeld (2004) and Hamdan v. Rumsfeld (2006). The Supreme Court has consistently held that even broad national security authorizations cannot override specific constitutional protections without explicit congressional authorization and appropriate procedural safeguards.

The verification also reveals I should consider whether the AI surveillance program has appropriate oversight mechanisms. The FISA framework provides a model for balancing national security needs with constitutional protections through judicial oversight, which appears absent from the described program.

Verified National Security Conclusion: While Congress possesses broad authority to authorize surveillance for national security purposes, such authorization cannot override specific constitutional protections without explicit authorization and appropriate procedural safeguards. The described program lacks the judicial oversight and individualized suspicion requirements that courts have required even in national security contexts.

Balancing Test and Final Constitutional Assessment:
Applying strict scrutiny analysis, the government must demonstrate that bulk AI surveillance serves compelling state interests and is narrowly tailored to achieve those interests using the least restrictive means available.

Verification Check #4: Let me verify my application of strict scrutiny by ensuring I've properly analyzed each prong of the test and considered whether a different standard of review might apply in the national security context.

Upon verification, I recognize that national security cases sometimes apply intermediate scrutiny rather than strict scrutiny, particularly when foreign intelligence gathering is involved. However, this case involves domestic surveillance of American citizens, which typically receives strict scrutiny protection under both Fourth and First Amendment analysis.

The compelling interest prong is likely satisfied - national security and terrorism prevention constitute compelling government interests. However, the narrow tailoring analysis is more problematic. The government must demonstrate that less restrictive alternatives (such as individualized warrants, targeted surveillance, or enhanced FISA procedures) would be inadequate to achieve the security objectives.

The program's targeting of 2.3 million citizens without individualized suspicion suggests over-breadth that would likely fail narrow tailoring analysis. The fact that the system flagged political dissidents and journalists based on communication content rather than specific security threats further undermines the tailoring argument.

Verified Final Constitutional Conclusion: The bulk AI surveillance program likely violates both Fourth and First Amendment protections because, while serving compelling national security interests, it is not narrowly tailored to achieve those interests and employs means that are broader than necessary given available alternatives such as targeted surveillance under FISA or other existing legal frameworks.

Therefore, this self-verifying constitutional analysis demonstrates that systematic verification of legal reasoning enhances accuracy by forcing reconsideration of initial conclusions, identification of overlooked precedent, and more thorough application of relevant legal standards, ultimately leading to more robust and defensible legal analysis.
""",
        "params": {
            "max_tokens": 4500,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 27: Self-Checking Scientific Analysis (Error Detection Loops)",
        "prompt": """
CONTEXT:
A pharmaceutical research team has developed a new drug compound that shows promising results in initial trials, but the data analysis requires systematic self-verification to catch potential errors, biases, and methodological flaws before proceeding to Phase III trials.

Experimental Data and Initial Results:
The new compound, designated XK-47, was tested in a randomized, double-blind, placebo-controlled Phase II trial involving 847 patients with moderate to severe hypertension. Primary endpoint was reduction in systolic blood pressure after 12 weeks of treatment. Initial analysis shows mean systolic BP reduction of 18.7 mmHg in treatment group vs 3.2 mmHg in placebo group (p < 0.001). Secondary endpoints include diastolic BP reduction (12.4 mmHg vs 1.8 mmHg, p < 0.001), cardiovascular events (3.2% vs 5.7%, p = 0.08), and adverse events (reported in 23% of treatment group vs 18% of placebo group, p = 0.15).

Study Design and Methodology:
Participants were randomized 2:1 to treatment vs placebo, with 565 receiving XK-47 and 282 receiving placebo. Inclusion criteria: ages 35-75, systolic BP 140-180 mmHg, no prior cardiovascular events. Exclusion criteria: pregnancy, kidney disease, liver disease, current use of more than two antihypertensive medications. Primary analysis used intention-to-treat principle. Dropout rate was 12% in treatment group and 8% in placebo group.

SCIENTIFIC ANALYSIS WITH VERIFICATION PROTOCOLS:
This critical pharmaceutical analysis requires systematic evaluation followed by mandatory self-verification loops to identify potential errors, biases, and analytical flaws before making recommendations for Phase III trials.

Initial Statistical Analysis:
The primary endpoint shows statistically significant and clinically meaningful reduction in systolic blood pressure. The 18.7 mmHg reduction exceeds the 10 mmHg threshold typically considered clinically significant for cardiovascular benefit. The p-value of < 0.001 indicates very low probability that results occurred by chance, and the effect size appears substantial.

Self-Verification Loop #1: Let me systematically check my initial statistical interpretation for potential errors or overlooked considerations.

Upon self-checking, I identify several concerns with my initial analysis:

First, I need to verify the clinical significance threshold. While 10 mmHg is often cited, the actual threshold depends on baseline risk factors and patient population. I should examine whether the 18.7 mmHg reduction is consistent across different patient subgroups and baseline BP levels.

Second, the p-value interpretation needs verification. While p < 0.001 suggests statistical significance, I must consider multiple comparisons, interim analyses, and whether the study was adequately powered for the observed effect size.

Third, I failed to adequately consider the dropout rate differential (12% vs 8%). This could introduce bias if dropouts were related to treatment effects or side effects.

Corrected Statistical Analysis: The primary endpoint results appear statistically significant, but I need additional analysis to verify clinical significance across patient subgroups and to assess whether the differential dropout rate affects the intention-to-treat analysis validity.

Safety Profile Assessment:
The adverse event rate of 23% in the treatment group vs 18% in placebo appears concerning, though not statistically significant (p = 0.15). This 5 percentage point difference could represent approximately 28 additional adverse events per 565 treated patients.

Self-Verification Loop #2: Let me check my safety analysis for potential errors in interpretation and ensure I haven't missed important safety signals.

Upon verification, I realize several critical omissions in my safety analysis:

First, I need to examine the types and severity of adverse events, not just overall rates. Some adverse events might be minor (headache, nausea) while others could be serious (cardiovascular events, liver toxicity).

Second, the p-value of 0.15 doesn't necessarily indicate safety. The confidence interval around the difference should be calculated to understand the range of possible increased risk.

Third, I failed to consider dose-response relationships for adverse events and whether certain patient subgroups experienced higher rates.

Fourth, the 5 percentage point difference in adverse events could be clinically significant even if not statistically significant, particularly for serious events.

Corrected Safety Assessment: The adverse event profile requires detailed analysis by event type and severity. A 5 percentage point increase in adverse events could represent clinically significant safety concerns that need thorough evaluation before Phase III trials.

Efficacy vs Safety Risk-Benefit Analysis:
Balancing the 18.7 mmHg BP reduction against a potential 5% increase in adverse events requires careful consideration of the types of events and long-term implications.

Self-Verification Loop #3: Let me verify my risk-benefit analysis by checking for potential biases and ensuring I'm using appropriate analytical frameworks.

Upon self-checking, I identify several issues with my risk-benefit approach:

First, I'm comparing short-term adverse events (12 weeks) with long-term cardiovascular benefits that may accrue over years. This temporal mismatch requires adjustment using appropriate cardiovascular risk prediction models.

Second, I haven't considered whether the adverse events are reversible upon discontinuation or represent permanent harm.

Third, my analysis lacks comparison to existing antihypertensive medications. XK-47's risk-benefit profile should be compared to established treatments, not just placebo.

Fourth, I should consider whether the study population is representative of the broader hypertensive population who would receive this medication in clinical practice.

Corrected Risk-Benefit Analysis: A proper risk-benefit assessment requires long-term cardiovascular outcome modeling, comparison with existing treatments, and consideration of the reversibility of observed adverse events. The 12-week data provides limited insight into long-term risk-benefit ratios.

Study Design Validity Assessment:
The randomized, double-blind, placebo-controlled design represents the gold standard for pharmaceutical trials, and the sample size appears adequate for detecting clinically meaningful differences.

Self-Verification Loop #4: Let me systematically verify the study design validity and check for potential methodological flaws that could compromise the results.

Upon verification, I discover several design concerns I initially overlooked:

First, the 2:1 randomization ratio (treatment:placebo) may have been chosen to increase the treatment group size for safety assessment, but this reduces statistical power compared to 1:1 randomization.

Second, the exclusion criteria may have created a study population that doesn't represent typical hypertensive patients seen in clinical practice. Excluding patients with kidney disease, liver disease, or multiple medications could limit generalizability.

Third, the 12-week duration may be too short to assess long-term efficacy and safety. Blood pressure medications often show different risk-benefit profiles with longer-term use.

Fourth, I should verify whether the study had appropriate stopping rules for safety and whether any interim analyses were performed that could affect the statistical interpretation.

Corrected Study Design Assessment: While the basic design is sound, several factors including the 2:1 randomization, restrictive exclusion criteria, and short duration may limit the generalizability and long-term predictive value of the results.

Phase III Trial Recommendations:
Based on the verified analysis, XK-47 shows promising efficacy but requires additional safety evaluation and longer-term assessment.

Self-Verification Loop #5: Let me verify my Phase III recommendations by ensuring they address all identified concerns and follow appropriate regulatory and scientific standards.

Upon final verification, my Phase III recommendations should address:

1. Longer study duration (at least 6-12 months) to better assess long-term safety and sustained efficacy
2. More inclusive patient population to improve generalizability
3. Active comparator arm with established antihypertensive medication, not just placebo
4. Detailed cardiovascular outcome assessment, not just blood pressure reduction
5. Enhanced safety monitoring with predefined stopping rules for adverse events
6. Pharmacokinetic and pharmacodynamic studies in special populations (elderly, renal impairment, hepatic impairment)

Therefore, this self-verifying scientific analysis demonstrates that systematic verification loops significantly improve analytical rigor by identifying overlooked methodological concerns, statistical interpretation errors, and design limitations that could lead to inappropriate conclusions about drug safety and efficacy.
""",
        "params": {
            "max_tokens": 5000,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 28: Self-Auditing Financial Analysis (Assumption Verification)",
        "prompt": """
CONTEXT:
A private equity firm is considering a $2.3 billion acquisition of a technology company, but the financial analysis requires systematic self-verification to identify potential errors in assumptions, valuation methodologies, and risk assessments before presenting recommendations to the investment committee.

Target Company Profile:
CloudTech Solutions is a B2B software-as-a-service company providing enterprise data analytics platforms to Fortune 500 companies. Current annual recurring revenue (ARR) of $485 million with 78% gross margins, serving 1,247 enterprise clients across 23 countries. The company has grown ARR by 35% annually over the past three years, with net revenue retention of 118% and gross revenue retention of 94%. Current EBITDA margins are 22% with significant investment in sales and marketing (45% of revenue) and R&D (28% of revenue).

Market and Competitive Analysis:
The enterprise analytics market is projected to grow at 12% CAGR over the next five years, driven by increased demand for data-driven decision making and regulatory compliance requirements. CloudTech faces competition from established players like Tableau, Microsoft Power BI, and emerging AI-powered analytics platforms. The company's competitive advantages include proprietary machine learning algorithms, strong customer relationships, and comprehensive data integration capabilities.

FINANCIAL ANALYSIS WITH SELF-VERIFICATION:
This critical investment analysis requires thorough financial evaluation followed by systematic self-verification loops to identify potential errors in assumptions and methodologies before making acquisition recommendations.

Initial Valuation Analysis:
Using a discounted cash flow (DCF) model with 10-year projections, I assume revenue growth declining from 35% to 15% by year 5, EBITDA margins expanding to 35% by year 10, and WACC of 9.5%. The terminal growth rate assumption is 3%. Initial DCF valuation yields $2.8 billion enterprise value.

Self-Verification Loop #1: Let me systematically verify my DCF assumptions and methodology for potential errors or unrealistic projections.

Upon self-checking, I identify several concerning assumptions in my initial valuation:

Revenue Growth Assumptions: My projection of 35% declining to 15% may be too optimistic. I need to verify this against comparable company analysis and market size constraints. The enterprise analytics market at 12% CAGR suggests that sustaining 15% growth long-term would require significant market share gains.

EBITDA Margin Expansion: My assumption of margins expanding from 22% to 35% requires verification. This implies significant operating leverage, but I need to assess whether fixed costs are truly fixed and if competitive pressures might limit pricing power.

WACC Calculation: I should verify my 9.5% WACC by breaking down cost of equity and debt components. For a high-growth SaaS company, the cost of equity may be higher due to execution risk and market volatility.

Terminal Growth Rate: The 3% terminal growth rate may be too high for a technology company that could face disruption over a 10-year horizon.

Revised DCF Analysis: After verification, I need more conservative assumptions: revenue growth declining to 8-10% by year 5 (closer to market growth), EBITDA margins expanding to 28-30% (more realistic given competitive pressures), WACC of 10.5-11% (reflecting higher risk), and terminal growth rate of 2-2.5%.

Comparable Company Analysis:
Initial comparable analysis using trading and transaction multiples shows CloudTech trading at a discount to high-growth SaaS peers. Average EV/Revenue multiple for comparable companies is 12x, suggesting fair value of $5.8 billion based on current ARR.

Self-Verification Loop #2: Let me verify my comparable company analysis for potential selection bias and methodological errors.

Upon verification, I identify several issues with my comparables analysis:

Comparable Selection: I need to verify that my chosen comparables are truly similar in terms of growth rates, profitability, market served, and business model. Many high-multiple SaaS companies may have different risk profiles or growth trajectories.

Multiple Selection: EV/Revenue multiples can be misleading for companies at different profitability levels. I should also examine EV/EBITDA multiples and growth-adjusted metrics like PEG ratios.

Market Conditions: The timing of comparable transactions matters. Recent market volatility may have affected SaaS valuations, making historical multiples less relevant.

Size Premium/Discount: CloudTech's size relative to public comparables may warrant a liquidity discount or control premium adjustment.

Revised Comparable Analysis: After verification, I should use a more nuanced approach with multiple valuation metrics, adjust for differences in growth and profitability, and apply appropriate discounts for size and market conditions. This suggests a valuation range of $1.8-2.4 billion rather than $5.8 billion.

Risk Assessment and Sensitivity Analysis:
Initial risk analysis identifies key risks including customer concentration (top 10 customers represent 34% of ARR), competitive threats from Microsoft and Google, technology obsolescence risk, and key personnel dependence.

Self-Verification Loop #3: Let me verify my risk assessment for completeness and ensure I haven't overlooked material risks that could affect valuation.

Upon verification, I realize several additional risks not adequately considered:

Regulatory Risk: I failed to assess potential data privacy and security regulation changes that could affect the business model or require significant compliance investment.

Economic Sensitivity: I need to verify how recession-resistant the business model is. B2B software spending may decline during economic downturns, affecting renewal rates and new customer acquisition.

Integration Risk: I haven't adequately assessed the complexity and risks of integrating CloudTech with the private equity firm's other portfolio companies or potential synergies.

Currency and Geographic Risk: With 23-country operations, I should assess foreign exchange exposure and geopolitical risks.

Technology Risk: The pace of AI and machine learning advancement could obsolete current technology platforms faster than anticipated.

Revised Risk Assessment: After verification, the risk profile appears higher than initially assessed, particularly regarding regulatory changes, economic sensitivity, and technology obsolescence. This suggests higher required returns and more conservative growth assumptions.

Financing Structure Analysis:
Initial financing plan assumes 60% debt, 40% equity financing, with debt at 5.5% interest rate and 7-year amortization schedule.

Self-Verification Loop #4: Let me verify the financing assumptions against current market conditions and the target company's cash flow capacity.

Upon verification, several financing concerns emerge:

Debt Capacity: I need to verify whether CloudTech's cash flows can support 60% leverage. SaaS companies often have variable cash flows that may not support high leverage ratios.

Interest Rate Environment: Current interest rates may be higher than my 5.5% assumption, particularly for leveraged acquisition financing.

Covenant Risk: I should assess potential debt covenant restrictions that could limit operational flexibility or require additional equity injections.

Refinancing Risk: Seven-year debt maturity may create refinancing risk if market conditions deteriorate or company performance falls short of projections.

Revised Financing Structure: After verification, a more conservative 40% debt, 60% equity structure may be appropriate, with higher assumed interest rates and more flexible covenant terms.

Investment Recommendation Synthesis:
Based on verified analysis, CloudTech appears overvalued at $2.3 billion asking price given revised DCF valuation of $1.6-2.1 billion and risk-adjusted comparable analysis of $1.8-2.4 billion.

Self-Verification Loop #5: Let me verify my final recommendation by ensuring all analytical components are consistent and the recommendation addresses investment committee requirements.

Upon final verification, I should ensure my recommendation addresses:

Strategic Fit: How does CloudTech align with the fund's investment thesis and portfolio strategy?
Exit Strategy: What are realistic exit scenarios and timelines given market conditions?
Value Creation Plan: What specific operational improvements or synergies justify the investment?
Downside Protection: What is the worst-case scenario and how can downside risk be mitigated?

Final Verified Recommendation: CloudTech represents an attractive business with strong fundamentals, but the $2.3 billion asking price appears 15-20% above fair value based on verified analysis. Recommend either negotiating a lower price ($1.9-2.0 billion) or passing on the opportunity unless management can demonstrate additional value creation opportunities not reflected in current projections.

Therefore, this self-verifying financial analysis demonstrates that systematic verification loops are essential for identifying overoptimistic assumptions, methodological errors, and overlooked risks that could lead to value-destructive investment decisions in private equity transactions.
""",
        "params": {
            "max_tokens": 5500,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 29: Self-Correcting Engineering Calculation (Error Detection Protocol)",
        "prompt": """
CONTEXT:
A structural engineering firm is designing a 45-story mixed-use skyscraper in a seismically active region. The critical structural calculations must include systematic self-verification loops to catch potentially catastrophic errors before construction begins, as calculation mistakes could result in building collapse and loss of life.

Building Specifications and Design Requirements:
The proposed tower will be 615 feet tall with 45 floors: 5 below-grade parking levels, ground floor retail, 15 floors of office space, and 24 floors of residential units. The building footprint is 120 feet × 180 feet with a reinforced concrete core and steel moment frame construction. Design loads include: dead load averaging 85 psf, live loads of 50 psf (office) and 40 psf (residential), wind loads based on 90 mph basic wind speed, and seismic loads for Seismic Design Category D with site class C soil conditions.

Structural System and Load Path:
The lateral force resistance system consists of a reinforced concrete core (30 feet × 50 feet) combined with perimeter steel moment frames. Gravity loads are carried by steel columns and beams with composite metal deck slabs. Foundation system includes 84 drilled piers extending 65 feet to bedrock, each designed for 2,200 kip capacity.

STRUCTURAL CALCULATION WITH SELF-VERIFICATION:
This critical life-safety analysis requires systematic structural calculations followed by mandatory verification loops to identify potential errors that could compromise building safety.

Initial Seismic Force Calculation:
Using ASCE 7 procedures for Seismic Design Category D, I calculate the design base shear using: V = CsW, where Cs is the seismic response coefficient and W is the effective seismic weight.

Building weight calculation: W = 85 psf × 21,600 sf/floor × 45 floors = 82,620 kips
Site-modified spectral accelerations: SMS = 1.2g, SM1 = 0.6g
Design spectral accelerations: SDS = 0.8g, SD1 = 0.4g
Fundamental period: Ta = Ct × hn^x = 0.028 × 615^0.8 = 3.85 seconds
Seismic response coefficient: Cs = SDS / (R/I) = 0.8 / (8/1.0) = 0.10
Design base shear: V = 0.10 × 82,620 = 8,262 kips

Self-Verification Loop #1: Let me systematically verify my seismic force calculations for potential errors in code application and computational mistakes.

Upon self-checking, I identify several critical errors in my initial seismic analysis:

Building Weight Error: I calculated weight using total building area including below-grade levels, but seismic weight should exclude below-grade levels that don't contribute to seismic forces. Corrected weight: W = 85 psf × 21,600 sf × 40 floors = 73,440 kips.

Period Calculation Error: I used the approximate period formula, but for buildings over 12 stories in Seismic Design Category D, a more detailed analysis is required. The concrete core significantly affects the building period, requiring consideration of both flexural and shear deformations.

Response Modification Factor: I used R = 8 for steel moment frames, but the dual system of concrete core plus steel frames may qualify for R = 8 for the special steel moment frame system, provided the steel frame carries at least 25% of the seismic force.

Seismic Response Coefficient Limits: I need to verify that Cs doesn't exceed maximum values: Cs max = SD1/(T × R/I) and Cs min = 0.044 × SDS × I.

Corrected Seismic Calculation: With W = 73,440 kips and refined period analysis (T ≈ 2.8 seconds), Cs = min(0.10, 0.4/(2.8×8)) = min(0.10, 0.018) = 0.044 (using minimum), V = 0.044 × 73,440 = 3,231 kips.

Foundation Design Analysis:
With corrected seismic forces, I need to verify foundation capacity. Total vertical load = 73,440 kips dead + live load reactions. With 84 piers at 2,200 kips capacity each, total capacity = 184,800 kips, providing adequate vertical load capacity.

Overturning moment from seismic forces: M = V × (0.75 × hn) = 3,231 × (0.75 × 615) = 1,490,000 kip-ft
Foundation plan dimensions: 120 ft × 180 ft
Restoring moment from dead load: MR = W × (180/2) = 73,440 × 90 = 6,609,600 kip-ft
Overturning safety factor = 6,609,600 / 1,490,000 = 4.4 > 1.5 ✓

Self-Verification Loop #2: Let me verify my foundation analysis for potential errors in load combinations and stability calculations.

Upon verification, I discover significant errors in my foundation analysis:

Load Combination Error: I only considered seismic overturning, but I need to check all critical load combinations including wind loads. Wind may govern over seismic for this building height.

Foundation Configuration: I assumed all 84 piers contribute equally to overturning resistance, but piers near the center contribute less. I need to calculate the effective foundation width for overturning resistance.

Uplift Analysis: Under overturning conditions, some foundation elements may experience uplift forces. I need to verify that piers can resist tension forces or provide adequate tie-down.

Soil Bearing Pressure: I haven't verified that the foundation pressures don't exceed allowable soil bearing capacity under combined vertical and lateral loads.

Corrected Foundation Analysis: Wind loads may govern (need wind analysis), effective foundation resistance is less than total pier capacity, uplift forces require tension capacity verification, and soil bearing pressures need verification under all load combinations.

Steel Frame Design Check:
For the dual system, steel moment frames must carry at least 25% of seismic forces to qualify for R = 8. Required frame capacity = 0.25 × 3,231 = 808 kips per direction.

Assuming 6 moment frames in each direction, each frame carries 808/6 = 135 kips
Typical frame height = 615 ft, frame bay width = 30 ft
Frame drift limit = 615/400 = 1.54 inches (for office occupancy)

Self-Verification Loop #3: Let me verify my steel frame analysis for adequacy and proper load distribution between dual system components.

Upon verification, I find several issues with my steel frame analysis:

Load Distribution: The actual distribution between concrete core and steel frames depends on relative stiffness, not arbitrary assignment. I need rigorous analysis of system stiffness to determine load sharing.

Drift Analysis: I calculated total building drift limit, but need to verify that both steel frames and concrete core remain within their individual drift limits and compatibility.

Connection Design: Steel moment connections require special seismic detailing for Seismic Design Category D, including prequalified connections or special testing.

P-Delta Effects: For a 615-foot building, second-order P-Delta effects could be significant and must be included in the analysis.

Corrected Steel Frame Analysis: Requires rigorous analysis of dual system stiffness distribution, detailed drift analysis considering P-Delta effects, and verification that steel connections meet special seismic requirements.

Concrete Core Design Verification:
The reinforced concrete core provides the majority of lateral resistance. Core dimensions: 30 ft × 50 ft
Assuming the core carries 75% of seismic force: Vc = 0.75 × 3,231 = 2,423 kips
Core wall thickness assumed at 24 inches at base, tapering to 12 inches at top

Self-Verification Loop #4: Let me verify my concrete core design for adequacy under seismic loads and proper reinforcement detailing.

Upon verification, I identify critical omissions in my concrete core analysis:

Shear Wall Design: I need to verify that core walls can resist both in-plane and out-of-plane forces, including torsional effects due to mass eccentricity.

Boundary Element Design: Seismic Design Category D requires special boundary elements at wall ends if compression stress exceeds limits.

Reinforcement Detailing: Special seismic reinforcement requirements including confinement, splice requirements, and development lengths must be verified.

Foundation-Core Connection: The connection between core walls and foundation requires special design for force transfer and ductility.

Corrected Core Analysis: Requires detailed shear wall analysis, boundary element verification, special seismic reinforcement design, and foundation connection analysis.

Overall Structural System Verification:
The dual system must work together to provide adequate strength, stiffness, and ductility for life safety performance.

Self-Verification Loop #5: Let me perform a final systematic check of the overall structural system for consistency and adequacy.

Upon final verification, I identify system-level issues requiring resolution:

System Compatibility: Steel frames and concrete core must have compatible deformations under lateral loads to ensure proper load sharing.

Progressive Collapse Resistance: Building must be designed to resist progressive collapse through alternate load paths.

Construction Sequence: The construction sequence affects load distribution and must be considered in design.

Quality Assurance: Special inspection requirements for seismic design must be specified and enforced.

Final Verified Structural Assessment: The dual system appears feasible but requires comprehensive analysis including: refined seismic force calculations (corrected V = 3,231 kips), rigorous dual system analysis for load distribution, detailed foundation design including uplift and soil bearing verification, steel moment frame design with special seismic connections, concrete core design with special boundary elements and reinforcement, and systematic construction and inspection procedures.

Therefore, this self-verifying structural analysis demonstrates that systematic verification loops are essential for identifying calculation errors, code application mistakes, and design omissions that could lead to catastrophic structural failure, ultimately ensuring public safety through rigorous engineering practices.
""",
        "params": {
            "max_tokens": 6000,
            "temperature": 0.05,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 30: Self-Validating Strategic Analysis (Assumption Challenge Protocol)",
        "prompt": """
CONTEXT:
A Fortune 500 retail corporation is developing a five-year strategic transformation plan to compete against Amazon and other e-commerce giants. The strategic analysis must include systematic self-validation loops to challenge assumptions, test strategic logic, and identify potential blind spots that could lead to strategic failure.

Company Current Position:
RetailMax operates 2,847 physical stores across North America with annual revenue of $47 billion, primarily from in-store sales (78%) with growing online presence (22%). The company faces declining same-store sales (-2.3% annually over three years), market share erosion to e-commerce competitors, and changing consumer preferences toward omnichannel shopping experiences. Current competitive advantages include extensive physical footprint, established supplier relationships, private label capabilities, and strong brand recognition in core demographics.

Strategic Transformation Challenges:
Key challenges include: legacy IT infrastructure incompatible with modern e-commerce platforms, real estate footprint optimized for different consumer behaviors, workforce skills gap for digital transformation, supply chain designed for store replenishment rather than direct-to-consumer fulfillment, and organizational culture resistant to rapid change. The company must transform while maintaining current revenue streams during the transition period.

STRATEGIC ANALYSIS WITH SELF-VALIDATION:
This critical strategic transformation requires systematic analysis followed by rigorous self-validation loops to challenge assumptions and identify potential strategic blind spots before committing resources.

Initial Strategic Assessment:
RetailMax's core challenge is evolving from a store-centric to an omnichannel retailer while leveraging existing physical assets. The strategy should focus on: converting stores to fulfillment centers, implementing unified inventory management, developing robust e-commerce platform, creating seamless customer experience across channels, and optimizing real estate footprint for new operating model.

The transformation timeline is estimated at 5 years with total investment of $8.5 billion, expecting to achieve 60% online sales by year 5 and return to positive same-store sales growth by year 3.

Self-Validation Loop #1: Let me systematically challenge my initial strategic assessment by questioning underlying assumptions and testing the logic of the transformation approach.

Upon self-validation, I identify several questionable assumptions in my initial strategy:

Store-to-Fulfillment Center Assumption: I assumed physical stores can efficiently convert to fulfillment centers, but store locations are optimized for customer access, not logistics efficiency. Fulfillment centers need different infrastructure including loading docks, expanded storage, and automated sorting systems that may not be feasible in retail locations.

Timeline Assumption: My 5-year timeline may be unrealistic given the complexity of IT transformation, organizational change, and market competition. Amazon took decades to build their capabilities, and established retailers have additional legacy constraints.

Investment Return Assumption: The $8.5 billion investment assumes predictable returns, but digital transformation has high failure rates and uncertain outcomes. I need to assess what portion of this investment is at risk if the strategy fails.

Market Share Assumption: I assumed RetailMax can capture significant online market share, but this market is increasingly dominated by Amazon, and late entrants face significant disadvantage in customer acquisition costs and platform capabilities.

Revised Strategic Assessment: The transformation strategy faces higher risks and complexity than initially assessed. Success requires addressing fundamental questions about store economics in an omnichannel model, realistic transformation timelines, and competitive positioning in mature e-commerce markets.

Competitive Analysis and Market Position:
RetailMax competes against Amazon (dominant in e-commerce), Walmart (successful omnichannel transformation), Target (strong digital growth), and specialized online retailers. The company's differentiation must leverage unique assets while addressing competitive disadvantages.

Competitive advantages include: physical presence for product trial and immediate fulfillment, established customer relationships and loyalty programs, private label product capabilities, and potential for localized inventory and services.

Self-Validation Loop #2: Let me validate my competitive analysis by challenging whether identified advantages are sustainable and differentiated enough to compete effectively.

Upon validation, I question several competitive advantage assumptions:

Physical Presence Value: I assumed physical stores provide competitive advantage, but consumer behavior increasingly favors online convenience over in-store experience. The value of physical presence may continue declining faster than assumed.

Customer Loyalty Assumption: Established customer relationships may not translate to digital channels where switching costs are low and Amazon Prime creates powerful alternative loyalty.

Private Label Differentiation: Private label capabilities exist across retailers, and Amazon has aggressively expanded private label offerings with data advantages that traditional retailers can't match.

Localization Advantage: I assumed local inventory provides advantage, but Amazon's logistics network increasingly offers same-day delivery that may neutralize RetailMax's physical proximity benefits.

Revised Competitive Analysis: RetailMax's competitive advantages are less defensible than initially assumed. Success requires creating new sources of differentiation beyond traditional retail strengths, potentially through specialized customer experiences or market segments where physical presence retains premium value.

Financial Projections and Investment Analysis:
Initial financial modeling assumes: declining physical store revenue offset by growing online sales, margin compression during transformation followed by recovery, capital investment concentrated in years 1-3, and positive cash flow generation by year 4.

Key financial assumptions: online gross margins of 35% vs 42% for physical stores, customer acquisition costs of $67 per online customer, technology infrastructure costs of $2.3 billion over 5 years, and real estate optimization generating $1.2 billion in asset sales.

Self-Validation Loop #3: Let me validate my financial assumptions by challenging the economic logic and testing sensitivity to key variables.

Upon validation, I identify concerning financial assumptions:

Margin Assumption: I assumed online margins of 35%, but this may be optimistic given digital marketing costs, shipping expenses, and price competition online. Many retailers struggle to achieve online profitability comparable to physical stores.

Customer Acquisition Cost: $67 per customer assumes efficient digital marketing, but customer acquisition costs have increased significantly as digital advertising becomes more competitive and expensive.

Technology ROI: $2.3 billion technology investment assumes successful implementation and adoption, but enterprise IT transformations frequently exceed budgets and fail to deliver expected benefits.

Real Estate Value: $1.2 billion in asset sales assumes robust commercial real estate market, but retail property values may decline as more retailers downsize physical footprints.

Revenue Transition Risk: I assumed smooth transition from physical to online revenue, but channel conflict and execution challenges could result in revenue decline in both channels simultaneously.

Revised Financial Analysis: Financial projections are likely too optimistic. Success requires more conservative assumptions about margins, higher customer acquisition costs, technology implementation risks, and potential revenue decline during transformation.

Organizational Capability Assessment:
RetailMax must develop new capabilities including: advanced analytics and data science, digital marketing and e-commerce operations, supply chain optimization for omnichannel fulfillment, technology development and maintenance, and change management for cultural transformation.

Current organizational strengths include: retail operations expertise, merchandising and buying capabilities, supplier relationship management, and store management experience.

Self-Validation Loop #4: Let me validate the organizational transformation requirements by challenging whether the company can realistically develop needed capabilities within the strategic timeline.

Upon validation, I question several capability development assumptions:

Talent Acquisition: I assumed RetailMax can attract digital talent competitive with technology companies, but traditional retailers often struggle to recruit top digital talent due to cultural and compensation differences.

Cultural Change Speed: I assumed organizational culture can transform within 5 years, but retail culture emphasizing physical operations and customer service may resist digital-first approaches more than anticipated.

Technology Development vs. Acquisition: I assumed building internal capabilities, but RetailMax may need to acquire companies or form partnerships to access critical digital capabilities quickly enough.

Change Management Complexity: I underestimated the challenge of managing simultaneous changes across technology, operations, culture, and customer experience without disrupting current business.

Leadership Capability: I assumed existing leadership can guide digital transformation, but traditional retail executives may lack experience managing technology-driven business models.

Revised Organizational Assessment: Organizational transformation may be the most challenging aspect of the strategy, requiring significant external talent acquisition, potential acquisitions of digital capabilities, and realistic expectations about cultural change timelines.

Strategic Risk Assessment and Mitigation:
Key strategic risks include: competitive response from Amazon and other digital leaders, execution risks in technology implementation, customer defection during transformation period, financial pressures from declining margins and high investment requirements, and organizational resistance to change.

Risk mitigation strategies involve: phased implementation to limit exposure, partnerships with technology providers to reduce execution risk, customer retention programs during transition, and financial reserves to weather transformation period.

Self-Validation Loop #5: Let me conduct final validation of the overall strategic plan by challenging whether the strategy adequately addresses fundamental market realities and company constraints.

Upon final validation, I must address fundamental strategic questions:

Strategic Logic: Does RetailMax's transformation strategy differentiate sufficiently from competitors' approaches, or is it following a similar path that may lead to commoditization?

Market Timing: Is RetailMax entering digital transformation too late to compete effectively against established players with significant first-mover advantages?

Resource Requirements: Can RetailMax realistically fund the transformation while maintaining financial stability and shareholder support throughout the multi-year process?

Success Probability: Given the high failure rate of digital transformations and retail industry challenges, what is the realistic probability of achieving strategic objectives?

Alternative Strategies: Should RetailMax consider alternative strategies such as focusing on specific market segments, forming strategic alliances, or pursuing merger opportunities rather than independent transformation?

Final Validated Strategic Recommendation: While digital transformation is strategically necessary, RetailMax faces significant challenges that make independent transformation high-risk. The company should consider a hybrid approach combining selective digital capabilities development with strategic partnerships or acquisition targets that provide faster access to e-commerce expertise. The strategy should emphasize differentiated customer experiences and market segments where physical presence retains value, rather than attempting to replicate Amazon's broad e-commerce model.

Therefore, this self-validating strategic analysis demonstrates that systematic assumption challenging is essential for identifying strategic blind spots, unrealistic expectations, and fundamental market realities that could lead to strategic failure if not adequately addressed in planning and execution.
""",
        "params": {
            "max_tokens": 6500,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 31: Multi-Variable Probability Analysis (Bayesian Reasoning)",
        "prompt": """
CONTEXT:
A complex medical diagnostic system uses multiple interdependent tests to diagnose a rare genetic condition. The system must calculate precise probabilities using Bayesian reasoning, considering test accuracies, disease prevalence, and conditional dependencies between multiple diagnostic markers.

Medical Diagnostic Scenario:
Genomix syndrome is a rare genetic disorder affecting 0.08% of the population (base rate = 0.0008). The diagnostic protocol involves five independent tests, each with different accuracy characteristics and cost considerations. Medical decisions require precise probability calculations to determine optimal testing sequences and treatment recommendations.

Test Accuracy Parameters:
Test A (Genetic Marker Analysis): Sensitivity = 0.94 (correctly identifies 94% of affected individuals), Specificity = 0.97 (correctly identifies 97% of unaffected individuals), Cost = $850

Test B (Enzyme Level Assessment): Sensitivity = 0.89, Specificity = 0.93, Cost = $340, Note: Results can be influenced by patient age and certain medications

Test C (Protein Expression Profile): Sensitivity = 0.91, Specificity = 0.95, Cost = $720, Note: Requires 48-hour processing time

Test D (Metabolite Concentration): Sensitivity = 0.87, Specificity = 0.89, Cost = $280, Note: Must be performed on fasting patients

Test E (Cellular Function Assay): Sensitivity = 0.96, Specificity = 0.98, Cost = $1,200, Note: Most accurate but expensive and technically demanding

BAYESIAN PROBABILITY CALCULATION ANALYSIS:
This complex diagnostic analysis requires systematic probability calculations using Bayes' theorem and conditional probability to determine optimal testing strategies and diagnostic confidence levels.

Initial Single-Test Probability Analysis:
For any single test, the probability of having the disease given a positive result is calculated using Bayes' theorem: P(Disease|Positive Test) = P(Positive Test|Disease) × P(Disease) / P(Positive Test)

Test A Analysis:
P(Genomix|Positive A) = (0.94 × 0.0008) / [0.94 × 0.0008 + (1-0.97) × (1-0.0008)]
P(Genomix|Positive A) = 0.000752 / [0.000752 + 0.03 × 0.9992]
P(Genomix|Positive A) = 0.000752 / [0.000752 + 0.029976]
P(Genomix|Positive A) = 0.000752 / 0.030728 = 0.0245 or 2.45%

This counterintuitive result demonstrates that even with a highly accurate test, the low disease prevalence means that most positive results are false positives. The posterior probability of having the disease after a single positive Test A is only 2.45%.

Test E Analysis (Most Accurate):
P(Genomix|Positive E) = (0.96 × 0.0008) / [0.96 × 0.0008 + (1-0.98) × (1-0.0008)]
P(Genomix|Positive E) = 0.000768 / [0.000768 + 0.02 × 0.9992]
P(Genomix|Positive E) = 0.000768 / [0.000768 + 0.019984]
P(Genomix|Positive E) = 0.000768 / 0.020752 = 0.0370 or 3.70%

Even the most accurate and expensive test yields only 3.70% probability after a positive result, highlighting the challenge of diagnosing rare conditions.

Sequential Testing Analysis:
When tests are performed sequentially, each result updates the prior probability for subsequent tests. If Test A is positive (posterior probability = 2.45%), this becomes the prior probability for Test B.

Test A Positive, Then Test B:
New prior probability = 0.0245
P(Genomix|A+,B+) = (0.89 × 0.0245) / [0.89 × 0.0245 + (1-0.93) × (1-0.0245)]
P(Genomix|A+,B+) = 0.021805 / [0.021805 + 0.07 × 0.9755]
P(Genomix|A+,B+) = 0.021805 / [0.021805 + 0.068285]
P(Genomix|A+,B+) = 0.021805 / 0.09009 = 0.242 or 24.2%

After two positive tests, the probability increases substantially to 24.2%. This demonstrates the power of sequential testing for rare conditions.

Triple Positive Results Analysis:
Continuing with Test C after A+ and B+:
New prior probability = 0.242
P(Genomix|A+,B+,C+) = (0.91 × 0.242) / [0.91 × 0.242 + (1-0.95) × (1-0.242)]
P(Genomix|A+,B+,C+) = 0.22022 / [0.22022 + 0.05 × 0.758]
P(Genomix|A+,B+,C+) = 0.22022 / [0.22022 + 0.0379]
P(Genomix|A+,B+,C+) = 0.22022 / 0.25812 = 0.853 or 85.3%

After three positive tests, diagnostic confidence reaches 85.3%, approaching clinical decision thresholds.

Negative Test Result Impact Analysis:
What happens if Test A is positive but Test B is negative?
P(Genomix|A+,B-) = (0.11 × 0.0245) / [0.11 × 0.0245 + 0.93 × (1-0.0245)]
P(Genomix|A+,B-) = 0.002695 / [0.002695 + 0.907215]
P(Genomix|A+,B-) = 0.002695 / 0.90991 = 0.00296 or 0.296%

A negative Test B after positive Test A dramatically reduces the probability to 0.296%, demonstrating how contradictory evidence affects diagnostic confidence.

Optimal Testing Strategy Analysis:
Given cost and accuracy considerations, what is the optimal testing sequence to achieve 95% diagnostic confidence while minimizing cost?

Strategy 1: Start with most accurate test (E), then add others as needed
- Test E positive → 3.70% probability, cost = $1,200
- Add Test A → probability increases to approximately 20.8%, total cost = $2,050
- Add Test C → probability increases to approximately 83.2%, total cost = $2,770
- Add Test B → probability increases to approximately 95.1%, total cost = $3,110

Strategy 2: Start with least expensive tests, add expensive ones as needed
- Test D positive → approximately 2.1% probability, cost = $280
- Add Test B → approximately 15.3% probability, total cost = $620
- Add Test C → approximately 67.8% probability, total cost = $1,340
- Add Test A → approximately 93.2% probability, total cost = $2,190
- Add Test E → probability exceeds 99%, total cost = $3,390

Expected Value Analysis:
Considering the probability of needing each additional test and associated costs:

Expected cost for Strategy 1 = $1,200 + 0.963 × $850 + 0.928 × $720 + 0.847 × $340 = $3,975
Expected cost for Strategy 2 = $280 + 0.979 × $340 + 0.943 × $720 + 0.756 × $850 + 0.512 × $1,200 = $3,289

Strategy 2 (starting with cheaper tests) has lower expected cost while achieving comparable diagnostic accuracy.

False Discovery Rate Analysis:
In a population of 100,000 people, approximately 80 would have Genomix syndrome. Using optimal Strategy 2:
- True positives reaching 95% threshold: approximately 76 patients
- False positives reaching 95% threshold: approximately 3-4 patients
- False discovery rate: 4-5%
- Number needed to test to find one true positive: approximately 1,316

Cost-Effectiveness Analysis:
Total cost to identify all cases in 100,000 population using Strategy 2: approximately $263,280
Cost per true positive identified: approximately $3,464
Cost per quality-adjusted life year (assuming early treatment adds 15 QALYs): approximately $231

Therefore, this comprehensive Bayesian analysis demonstrates that systematic probability calculations are essential for optimizing diagnostic strategies in rare disease scenarios, revealing counterintuitive results where highly accurate individual tests have low positive predictive value, sequential testing dramatically improves diagnostic confidence, and cost-effective strategies require balancing test accuracy against expense through expected value analysis.
""",
        "params": {
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 32: Multi-Constraint Optimization Problem (Linear Programming Logic)",
        "prompt": """
CONTEXT:
A manufacturing company must optimize production across five product lines while satisfying multiple constraints including resource limitations, demand requirements, quality standards, and regulatory compliance. The optimization requires systematic mathematical reasoning using linear programming principles.

Manufacturing Optimization Challenge:
TechManufacturing Inc. produces five high-tech components (Products A, B, C, D, E) used in aerospace, automotive, and electronics industries. Each product requires different combinations of raw materials, labor hours, machine time, and quality control processes. The company must determine optimal production quantities to maximize profit while satisfying all operational constraints.

Resource Constraint Parameters:
Raw Material Constraints:
- Titanium alloy: 2,400 kg available monthly
  Product A requires 8 kg, B requires 12 kg, C requires 0 kg, D requires 15 kg, E requires 6 kg
- Carbon fiber: 1,800 kg available monthly  
  Product A requires 0 kg, B requires 5 kg, C requires 18 kg, D requires 8 kg, E requires 12 kg
- Specialty polymers: 3,200 kg available monthly
  Product A requires 15 kg, B requires 0 kg, C requires 22 kg, D requires 0 kg, E requires 28 kg

Labor Hour Constraints:
- Skilled technicians: 4,800 hours available monthly
  Product A requires 12 hours, B requires 18 hours, C requires 8 hours, D requires 25 hours, E requires 15 hours
- Quality inspectors: 2,200 hours available monthly
  Product A requires 6 hours, B requires 8 hours, C requires 4 hours, D requires 12 hours, E requires 9 hours

Machine Time Constraints:
- CNC machining: 3,600 hours available monthly
  Product A requires 20 hours, B requires 15 hours, C requires 0 hours, D requires 35 hours, E requires 10 hours
- 3D printing: 2,800 hours available monthly
  Product A requires 0 hours, B requires 10 hours, C requires 25 hours, D requires 8 hours, E requires 18 hours

LINEAR PROGRAMMING MATHEMATICAL ANALYSIS:
This complex optimization problem requires systematic mathematical reasoning using linear programming methodology to determine optimal production quantities while satisfying all constraints.

Decision Variables Definition:
Let xA, xB, xC, xD, xE represent the number of units to produce of products A, B, C, D, E respectively.
All variables must be non-negative integers: xA, xB, xC, xD, xE ≥ 0

Objective Function Formulation:
Profit contributions per unit:
Product A: $2,400 revenue - $1,680 costs = $720 profit
Product B: $3,200 revenue - $2,240 costs = $960 profit  
Product C: $1,800 revenue - $1,260 costs = $540 profit
Product D: $4,500 revenue - $3,150 costs = $1,350 profit
Product E: $2,800 revenue - $1,960 costs = $840 profit

Maximize Z = 720xA + 960xB + 540xC + 1,350xD + 840xE

Constraint Equation System:
Raw Material Constraints:
Titanium: 8xA + 12xB + 0xC + 15xD + 6xE ≤ 2,400
Carbon fiber: 0xA + 5xB + 18xC + 8xD + 12xE ≤ 1,800
Polymers: 15xA + 0xB + 22xC + 0xD + 28xE ≤ 3,200

Labor Constraints:
Technicians: 12xA + 18xB + 8xC + 25xD + 15xE ≤ 4,800
Inspectors: 6xA + 8xB + 4xC + 12xD + 9xE ≤ 2,200

Machine Constraints:
CNC: 20xA + 15xB + 0xC + 35xD + 10xE ≤ 3,600
3D printing: 0xA + 10xB + 25xC + 8xD + 18xE ≤ 2,800

Demand and Quality Constraints:
Minimum demand requirements:
xA ≥ 15, xB ≥ 12, xC ≥ 20, xD ≥ 8, xE ≥ 18

Maximum production capacity:
xA ≤ 120, xB ≤ 90, xC ≤ 150, xD ≤ 60, xE ≤ 100

Constraint Analysis and Feasibility Assessment:
To solve this systematically, I need to identify binding constraints and potential corner points of the feasible region.

Resource Utilization Analysis:
If all products were produced at minimum demand levels (xA=15, xB=12, xC=20, xD=8, xE=18):

Titanium usage: 8(15) + 12(12) + 0(20) + 15(8) + 6(18) = 120 + 144 + 0 + 120 + 108 = 492 kg
Available: 2,400 kg, Utilization: 20.5%

Carbon fiber usage: 0(15) + 5(12) + 18(20) + 8(8) + 12(18) = 0 + 60 + 360 + 64 + 216 = 700 kg  
Available: 1,800 kg, Utilization: 38.9%

Polymer usage: 15(15) + 0(12) + 22(20) + 0(8) + 28(18) = 225 + 0 + 440 + 0 + 504 = 1,169 kg
Available: 3,200 kg, Utilization: 36.5%

At minimum production levels, no material constraints are binding. Labor and machine constraints must be checked similarly.

Profit Maximization Strategy:
Since Product D has the highest profit margin ($1,350), we should maximize xD first, then Product B ($960), then E ($840), then A ($720), and finally C ($540).

Testing Maximum Production of Product D:
If xD = 60 (maximum), resource requirements:
Titanium: 15(60) = 900 kg
Technicians: 25(60) = 1,500 hours  
Inspectors: 12(60) = 720 hours
CNC: 35(60) = 2,100 hours

Remaining resources after maximizing D:
Titanium: 2,400 - 900 = 1,500 kg remaining
Technicians: 4,800 - 1,500 = 3,300 hours remaining
Inspectors: 2,200 - 720 = 1,480 hours remaining
CNC: 3,600 - 2,100 = 1,500 hours remaining

Optimizing Remaining Products:
With xD = 60, optimize remaining products using leftover resources.
Next highest profit is Product B at $960 per unit.

Maximum B production with remaining resources:
Titanium constraint: 12xB ≤ 1,500 → xB ≤ 125, but max capacity is 90, so xB ≤ 90
Technician constraint: 18xB ≤ 3,300 → xB ≤ 183
CNC constraint: 15xB ≤ 1,500 → xB ≤ 100
Minimum demand: xB ≥ 12

Set xB = 90 (maximum capacity)

Remaining resources after D=60, B=90:
Titanium: 1,500 - 12(90) = 1,500 - 1,080 = 420 kg
Technicians: 3,300 - 18(90) = 3,300 - 1,620 = 1,680 hours
CNC: 1,500 - 15(90) = 1,500 - 1,350 = 150 hours

The CNC constraint becomes critical with only 150 hours remaining.

Final Product Allocation:
With remaining 150 CNC hours:
Product A requires 20 hours per unit → maximum 7 units (but minimum demand is 15)
Product E requires 10 hours per unit → maximum 15 units (minimum demand is 18)

This creates infeasibility - we cannot meet minimum demand requirements with remaining resources.

Revised Optimization Approach:
We need to reduce D and B production to satisfy all minimum demands.

Let me test: xD = 50, xB = 70, and check if minimum demands can be met.

Resource usage with xD=50, xB=70:
Titanium: 15(50) + 12(70) = 750 + 840 = 1,590 kg
Technicians: 25(50) + 18(70) = 1,250 + 1,260 = 2,510 hours
CNC: 35(50) + 15(70) = 1,750 + 1,050 = 2,800 hours

Remaining resources:
Titanium: 2,400 - 1,590 = 810 kg
Technicians: 4,800 - 2,510 = 2,290 hours  
CNC: 3,600 - 2,800 = 800 hours

For minimum demands A=15, C=20, E=18:
Titanium needed: 8(15) + 0(20) + 6(18) = 120 + 0 + 108 = 228 kg ✓
Technicians needed: 12(15) + 8(20) + 15(18) = 180 + 160 + 270 = 610 hours ✓
CNC needed: 20(15) + 0(20) + 10(18) = 300 + 0 + 180 = 480 hours ✓

This allocation satisfies all constraints.

Optimal Solution:
xA = 15, xB = 70, xC = 20, xD = 50, xE = 18

Total profit = 720(15) + 960(70) + 540(20) + 1,350(50) + 840(18) = 10,800 + 67,200 + 10,800 + 67,500 + 15,120 = $171,420

Resource utilization at optimal solution:
Titanium: 228 + 840 + 750 = 1,818 kg (75.8%)
CNC: 480 + 1,050 + 1,750 = 3,280 hours (91.1%)

Therefore, this systematic linear programming analysis demonstrates that mathematical optimization of complex multi-constraint systems requires careful constraint analysis, feasibility testing, and iterative refinement to balance profit maximization with operational requirements, ultimately yielding an optimal monthly production plan generating $171,420 profit while satisfying all resource, demand, and capacity constraints.
""",
        "params": {
            "max_tokens": 4500,
            "temperature": 0.05,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 33: Game Theory Nash Equilibrium Analysis (Strategic Logic)",
        "prompt": """
CONTEXT:
Five technology companies are competing in the emerging quantum computing market, and each must decide on research investment levels and patent strategies. The scenario requires game theory analysis to identify Nash equilibria and optimal competitive strategies under various market conditions.

Quantum Computing Market Competition:
Companies A (TechGiant), B (QuantumLeap), C (InnovateCorp), D (ResearchLabs), and E (StartupTech) are competing to dominate the quantum computing market. Each company must simultaneously decide on three strategic dimensions: R&D investment level (High, Medium, Low), patent strategy (Aggressive filing, Defensive portfolio, Open collaboration), and market timing (Early entry, Wait-and-see, Fast follower).

Company Characteristics and Capabilities:
Company A (TechGiant): Massive resources ($50B revenue), strong patent portfolio, risk-averse culture
Company B (QuantumLeap): Specialized quantum expertise, medium resources ($2B revenue), innovation-focused
Company C (InnovateCorp): Balanced capabilities, strong partnerships, moderate resources ($8B revenue)
Company D (ResearchLabs): Academic connections, limited resources ($800M revenue), cutting-edge research
Company E (StartupTech): Agile, venture-funded ($100M funding), high-risk tolerance

Strategic Payoff Analysis:
Each company's payoff depends on their own strategy choice and competitors' strategies. Payoffs represent expected net present value of quantum computing business over 10 years (in billions).

R&D Investment Decision Matrix:
If all companies choose High R&D investment:
- A: -$5B (overcrowded, high costs, diminishing returns)
- B: -$2B (resource strain, competitive pressure)  
- C: -$1B (stretched resources, moderate position)
- D: -$3B (unsustainable investment level)
- E: -$8B (bankruptcy risk, resource exhaustion)

If only one company chooses High R&D (assuming others choose Medium):
- Solo high investor gains significant competitive advantage
- A with High R&D vs others Medium: A = +$25B, others = -$2B each
- B with High R&D vs others Medium: B = +$15B, others = -$1B each
- Similar patterns for C (+$12B), D (+$8B), E (+$20B)

GAME THEORY NASH EQUILIBRIUM ANALYSIS:
This complex strategic interaction requires systematic game theory analysis to identify stable equilibria and optimal strategies for each competitor.

Pure Strategy Nash Equilibrium Search:
For a pure strategy Nash equilibrium, each player's strategy must be optimal given all other players' strategies. Let me analyze whether any pure strategy combinations are stable.

Scenario Analysis: All Choose High R&D Investment
If all companies choose (High R&D, Aggressive Patents, Early Entry):
Company A payoff: -$5B + patent premium -$2B + first-mover advantage +$8B = +$1B
Company B payoff: -$2B + patent premium -$1B + first-mover advantage +$6B = +$3B
Company C payoff: -$1B + patent premium -$1B + first-mover advantage +$4B = +$2B
Company D payoff: -$3B + patent premium -$0.5B + first-mover advantage +$2B = -$1.5B
Company E payoff: -$8B + patent premium -$0.5B + first-mover advantage +$3B = -$5.5B

This is not a Nash equilibrium because D and E would want to deviate to lower investment levels.

Mixed Strategy Equilibrium Analysis:
Since pure strategy equilibria appear unstable, I need to analyze mixed strategy equilibria where companies randomize over strategies.

Let p_i^H, p_i^M, p_i^L represent the probability that company i chooses High, Medium, or Low R&D investment respectively, where p_i^H + p_i^M + p_i^L = 1.

For Company A to be indifferent between High and Medium investment (mixed strategy condition):
Expected payoff from High = Expected payoff from Medium

E[π_A^H] = Σ(probability of competitor strategies) × (payoff to A from High given competitor strategies)
E[π_A^M] = Σ(probability of competitor strategies) × (payoff to A from Medium given competitor strategies)

This creates a system of indifference equations that must be solved simultaneously.

Simplified Two-Player Subgame Analysis:
To illustrate the methodology, consider just Companies A and B competing:

Company A's payoff matrix (rows) vs Company B's strategies (columns):
                B: High    B: Medium    B: Low
A: High         -2, -1      +15, -3      +20, -4
A: Medium       -3, +10     +8, +5       +12, +2  
A: Low          -4, +12     +2, +8       +5, +6

For mixed strategy equilibrium, A must be indifferent between strategies:
Let q = probability B chooses High, (1-r-q) = probability B chooses Medium, r = probability B chooses Low

A's expected payoff from High: -2q + 15(1-r-q) + 20r = 15 - 17q + 5r
A's expected payoff from Medium: -3q + 8(1-r-q) + 12r = 8 - 11q + 4r
A's expected payoff from Low: -4q + 2(1-r-q) + 5r = 2 - 6q + 3r

Setting High = Medium: 15 - 17q + 5r = 8 - 11q + 4r → 7 = 6q - r → r = 6q - 7
Setting Medium = Low: 8 - 11q + 4r = 2 - 6q + 3r → 6 = 5q - r → r = 5q - 6

Solving: 6q - 7 = 5q - 6 → q = 1, r = -1

This solution is invalid (negative probability), suggesting no mixed strategy equilibrium exists in this simplified case.

Multi-Dimensional Strategy Space Analysis:
The full game involves three dimensions: R&D investment, patent strategy, and market timing. Each combination creates different payoff structures.

Strategic Interdependencies:
1. R&D Investment affects technological capability and costs
2. Patent Strategy affects competitive positioning and licensing revenue
3. Market Timing affects first-mover advantages and competitive response

Combined Strategy Analysis:
Consider the strategy profile where:
- A chooses (Medium R&D, Defensive Patents, Wait-and-see)
- B chooses (High R&D, Aggressive Patents, Early Entry)  
- C chooses (Medium R&D, Open Collaboration, Fast Follower)
- D chooses (Low R&D, Open Collaboration, Wait-and-see)
- E chooses (High R&D, Aggressive Patents, Early Entry)

Payoff calculations for this profile:
Company A: Medium R&D cost -$2B, defensive patent value +$3B, wait-and-see timing +$1B = +$2B
Company B: High R&D cost -$4B, aggressive patent value +$8B, early entry advantage +$6B = +$10B
Company C: Medium R&D cost -$1.5B, collaboration benefits +$4B, fast follower advantage +$3B = +$5.5B
Company D: Low R&D cost -$0.5B, collaboration benefits +$2B, wait-and-see timing +$0.5B = +$2B
Company E: High R&D cost -$8B, aggressive patent value +$5B, early entry advantage +$4B = +$1B

Stability Analysis:
Is this a Nash equilibrium? Each player must check if unilateral deviation improves their payoff:

Company A deviation analysis:
- Switch to High R&D: Expected payoff becomes +$0B (worse)
- Switch to Low R&D: Expected payoff becomes +$1.5B (worse)
- Current strategy is optimal

Company B deviation analysis:
- Switch to Medium R&D: Expected payoff becomes +$8B (worse)
- Switch to Low R&D: Expected payoff becomes +$3B (worse)
- Current strategy is optimal

Similar analysis for other companies shows no beneficial unilateral deviations exist.

Nash Equilibrium Solution:
The strategy profile represents a Nash equilibrium:
- TechGiant (A): Medium R&D, Defensive Patents, Wait-and-see
- QuantumLeap (B): High R&D, Aggressive Patents, Early Entry  
- InnovateCorp (C): Medium R&D, Open Collaboration, Fast Follower
- ResearchLabs (D): Low R&D, Open Collaboration, Wait-and-see
- StartupTech (E): High R&D, Aggressive Patents, Early Entry

Market Outcome Implications:
This equilibrium results in:
- B and E emerge as technology leaders through high R&D investment
- A leverages resources efficiently with defensive strategy
- C benefits from collaboration while avoiding highest costs
- D survives through low-cost collaborative approach
- Industry structure: Two leaders (B, E), two followers (A, C), one niche player (D)

Therefore, this systematic game theory analysis demonstrates that strategic interactions in emerging technology markets require careful consideration of multidimensional strategy spaces, competitor capabilities, and payoff interdependencies, ultimately revealing stable competitive equilibria where companies pursue differentiated strategies based on their relative strengths and resource constraints.
""",
        "params": {
            "max_tokens": 5000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 34: Combinatorial Logic Puzzle (Constraint Satisfaction)",
        "prompt": """
CONTEXT:
An international mathematics competition presents a complex logic puzzle involving five teams from different countries, competing in five different mathematical disciplines, with results that must satisfy multiple logical constraints simultaneously. The puzzle requires systematic constraint satisfaction reasoning.

Mathematical Competition Logic Puzzle:
Five teams (Alpha, Beta, Gamma, Delta, Epsilon) from five countries (Brazil, Canada, France, Japan, Russia) compete in five mathematical disciplines (Algebra, Combinatorics, Geometry, Number Theory, Statistics). Each team consists of exactly three students, and the competition results must satisfy a complex set of logical constraints.

Competition Structure and Constraints:
Each discipline awards places 1st through 5th to the teams. Each team receives exactly one placement in each discipline, and each placement in each discipline goes to exactly one team.

Constraint Set 1 - Team Composition and Countries:
1. The Brazilian team (Alpha, Beta, Gamma, Delta, or Epsilon) placed 1st in exactly two disciplines
2. Team Alpha is not from Brazil, Canada, or Japan  
3. The Canadian team placed better than 3rd in Statistics but worse than 2nd in Number Theory
4. Team Beta is from either France or Russia
5. The Japanese team placed 5th in exactly one discipline and 1st in exactly one discipline
6. Team Gamma placed 2nd in Combinatorics
7. The Russian team did not place 5th in any discipline
8. Team Delta is not the Japanese team

Constraint Set 2 - Discipline-Specific Results:
9. In Algebra: Team Epsilon placed better than Team Alpha, but Team Delta placed better than Team Epsilon
10. In Combinatorics: The Brazilian team placed 4th
11. In Geometry: Exactly two teams from {Alpha, Beta, Gamma} placed in positions 1-3
12. In Number Theory: Team Beta placed 5th
13. In Statistics: The team that placed 1st in Statistics also placed 1st in exactly one other discipline
14. Team Alpha placed better in Geometry than in Algebra
15. The French team placed 3rd in exactly two disciplines

CONSTRAINT SATISFACTION LOGICAL ANALYSIS:
This complex combinatorial puzzle requires systematic constraint satisfaction reasoning to determine the unique solution that satisfies all given conditions simultaneously.

Initial Constraint Analysis and Deductions:
From constraints 2, 4, and 8: Team Alpha is from France or Russia (not Brazil, Canada, Japan). Team Beta is from France or Russia. Team Delta is not from Japan. This creates potential country assignment conflicts that must be resolved.

From constraint 6: Team Gamma placed 2nd in Combinatorics.
From constraint 10: The Brazilian team placed 4th in Combinatorics.
Since Team Gamma placed 2nd, Team Gamma is not the Brazilian team.

From constraint 12: Team Beta placed 5th in Number Theory.
From constraint 3: The Canadian team placed better than 3rd in Statistics (1st or 2nd) but worse than 2nd in Number Theory (3rd, 4th, or 5th).
Since Team Beta placed 5th in Number Theory, if Team Beta were Canadian, it would satisfy "worse than 2nd in Number Theory." We need to check Statistics placement.

Country Assignment Deduction Process:
From constraint 2: Alpha is from France or Russia
From constraint 4: Beta is from France or Russia  
From constraint 8: Delta is not from Japan
From constraint 7: The Russian team did not place 5th in any discipline

Since Team Beta placed 5th in Number Theory (constraint 12), and the Russian team never placed 5th (constraint 7), Team Beta cannot be Russian. Therefore, Team Beta must be French.

Since Team Beta is French and Alpha is from France or Russia (constraint 2), Team Alpha must be Russian.

From constraint 7: The Russian team (Alpha) never placed 5th in any discipline.
From constraint 15: The French team (Beta) placed 3rd in exactly two disciplines.

Discipline Placement Analysis:
In Number Theory: Team Beta (French) placed 5th
In Combinatorics: Team Gamma placed 2nd, Brazilian team placed 4th

From constraint 1: The Brazilian team placed 1st in exactly two disciplines.
Since the Brazilian team placed 4th in Combinatorics (not 1st), it must place 1st in two other disciplines.

From constraint 5: The Japanese team placed 5th in exactly one discipline and 1st in exactly one discipline.
From constraint 15: The French team (Beta) placed 3rd in exactly two disciplines.

Since Beta placed 5th in Number Theory, Beta must place 3rd in exactly two other disciplines to satisfy constraint 15.

From constraint 9: In Algebra, Epsilon > Alpha > Beta, meaning Delta > Epsilon > Alpha > Beta (since Delta > Epsilon).
Wait, let me re-read constraint 9: "Team Epsilon placed better than Team Alpha, but Team Delta placed better than Team Epsilon"
This means: Delta > Epsilon > Alpha in Algebra rankings (better = lower number/higher position).

Since Beta placed 5th in Number Theory and needs exactly two 3rd places, and we know from constraint 9 that in Algebra we have Delta > Epsilon > Alpha > [someone] > [someone], Beta cannot be in positions 1-3 in Algebra.

Let me systematically work through the Algebra constraint:
In Algebra: Delta > Epsilon > Alpha, and from constraint 14: Alpha placed better in Geometry than in Algebra.

If Alpha placed 4th in Algebra, then Alpha must place better than 4th (1st, 2nd, or 3rd) in Geometry.

Geometry Analysis:
From constraint 11: Exactly two teams from {Alpha, Beta, Gamma} placed in positions 1-3 in Geometry.
From constraint 14: Alpha placed better in Geometry than in Algebra.

Statistics Analysis:  
From constraint 3: The Canadian team placed better than 3rd in Statistics (1st or 2nd).
From constraint 13: The team that placed 1st in Statistics also placed 1st in exactly one other discipline.

Country Assignment Resolution:
We established: Alpha (Russian), Beta (French)
Remaining countries: Brazil, Canada, Japan for teams Gamma, Delta, Epsilon

From constraint 8: Delta is not Japanese
From constraint 5: The Japanese team placed 5th in exactly one discipline and 1st in exactly one discipline

Since we know the Brazilian team placed 4th in Combinatorics and 1st in exactly two disciplines (constraint 1), and the Japanese team placed 1st in exactly one discipline (constraint 5), these cannot be the same team.

From constraint 10: Brazilian team placed 4th in Combinatorics
From constraint 6: Gamma placed 2nd in Combinatorics
Therefore: Gamma is not Brazilian

Since Delta is not Japanese and Gamma is not Brazilian:
- If Delta is Brazilian, then Gamma and Epsilon are Canadian and Japanese (in some order)
- If Delta is Canadian, then Gamma is Japanese and Epsilon is Brazilian
- If Epsilon is Brazilian, then Gamma and Delta are Canadian and Japanese (in some order)

Testing Delta = Brazilian:
Delta (Brazilian) must place 1st in exactly two disciplines (constraint 1)
Delta (Brazilian) placed 4th in Combinatorics (constraint 10)
This is consistent.

From constraint 9: In Algebra, Delta > Epsilon > Alpha
If Delta placed 1st in Algebra, this could be one of Delta's two 1st places.

Testing remaining assignments with Delta (Brazilian):
Gamma and Epsilon are Canadian and Japanese.
From constraint 3: Canadian team placed 1st or 2nd in Statistics.
From constraint 5: Japanese team placed 5th in exactly one discipline and 1st in exactly one discipline.

If Gamma (2nd in Combinatorics) is Japanese, then Gamma must place 5th in exactly one discipline and 1st in exactly one discipline.
If Epsilon is Canadian, then Epsilon must place 1st or 2nd in Statistics.

Complete Solution Construction:
Working systematically through all constraints with Delta = Brazilian:

Teams and Countries: Alpha (Russian), Beta (French), Gamma (Japanese), Delta (Brazilian), Epsilon (Canadian)

Discipline placements:
- Combinatorics: [1st: ?], [2nd: Gamma], [3rd: ?], [4th: Delta], [5th: ?]
- Number Theory: [1st: ?], [2nd: ?], [3rd: ?], [4th: ?], [5th: Beta]
- Algebra: Delta > Epsilon > Alpha (positions 1, 2, 3), with Beta and Gamma in positions 4, 5
- Statistics: Epsilon (Canadian) in position 1 or 2
- Geometry: Exactly two of {Alpha, Beta, Gamma} in positions 1-3, with Alpha better than Alpha's Algebra position

Final Verified Solution:
Alpha (Russian): Algebra 3rd, Combinatorics 1st, Geometry 2nd, Number Theory 2nd, Statistics 4th
Beta (French): Algebra 4th, Combinatorics 5th, Geometry 3rd, Number Theory 5th, Statistics 3rd  
Gamma (Japanese): Algebra 5th, Combinatorics 2nd, Geometry 1st, Number Theory 1st, Statistics 5th
Delta (Brazilian): Algebra 1st, Combinatorics 4th, Geometry 5th, Number Theory 4th, Statistics 1st
Epsilon (Canadian): Algebra 2nd, Combinatorics 3rd, Geometry 4th, Number Theory 3rd, Statistics 2nd

Constraint Verification:
All 15 constraints satisfied:
✓ Brazilian (Delta) 1st in exactly 2 disciplines (Algebra, Statistics)
✓ Alpha not from Brazil, Canada, Japan (Russian)  
✓ Canadian (Epsilon) 2nd in Statistics, 3rd in Number Theory
✓ Beta from France or Russia (French)
✓ Japanese (Gamma) 5th in exactly 1 discipline, 1st in exactly 1 discipline
✓ Gamma 2nd in Combinatorics
✓ Russian (Alpha) never placed 5th
✓ Delta not Japanese (Brazilian)
✓ Algebra: Delta(1st) > Epsilon(2nd) > Alpha(3rd)
✓ Brazilian (Delta) 4th in Combinatorics
✓ Geometry: Alpha(2nd), Beta(3rd) from {Alpha,Beta,Gamma} in positions 1-3
✓ Beta 5th in Number Theory
✓ Statistics 1st (Delta) also 1st in Algebra
✓ Alpha better in Geometry(2nd) than Algebra(3rd)
✓ French (Beta) 3rd in exactly 2 disciplines (Geometry, Statistics)

Therefore, this systematic constraint satisfaction analysis demonstrates that complex combinatorial logic puzzles require methodical constraint propagation, systematic hypothesis testing, and comprehensive verification to identify the unique solution that simultaneously satisfies all logical requirements.
""",
        "params": {
            "max_tokens": 5500,
            "temperature": 0.05,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 35: Statistical Hypothesis Testing Chain (Mathematical Reasoning)",
        "prompt": """
CONTEXT:
A pharmaceutical research consortium is conducting a complex clinical trial comparing five different treatment protocols for a rare autoimmune disease. The statistical analysis requires multiple hypothesis tests with correction for multiple comparisons, analysis of covariance, and sophisticated interpretation of p-values and effect sizes.

Clinical Trial Statistical Analysis:
The IMMUNEX-5 trial enrolled 2,847 patients with Lambert-Eaton myasthenic syndrome across 47 international research centers. Patients were randomized to five treatment arms: (A) Standard immunosuppression, (B) Novel monoclonal antibody, (C) Plasma exchange protocol, (D) Combination therapy, (E) Experimental gene therapy. Primary endpoints include symptom improvement scores, quality of life measures, and biomarker levels at 6, 12, and 24 months.

Statistical Design Parameters:
Sample sizes per group: A (n=580), B (n=562), C (n=571), D (n=567), E (n=567)
Primary endpoint: Change in Quantitative Myasthenia Gravis (QMG) score from baseline to 24 months (lower scores indicate improvement)
Secondary endpoints: Quality of Life Scale (QoLS), Anti-VGCC antibody levels, Time to first clinical relapse
Statistical power: 90% to detect clinically meaningful difference of 3.2 points in QMG score
Alpha level: 0.05 (adjusted for multiple comparisons)

Baseline Characteristics and Potential Confounders:
Age distribution: Group A (mean 54.2, SD 14.8), Group B (mean 52.7, SD 15.3), Group C (mean 55.1, SD 14.2), Group D (mean 53.8, SD 15.7), Group E (mean 54.5, SD 14.9)
Disease duration: Group A (6.8 years), Group B (7.2 years), Group C (6.5 years), Group D (7.1 years), Group E (6.9 years)
Baseline QMG scores: Group A (18.4 ± 6.2), Group B (18.1 ± 6.4), Group C (18.7 ± 5.9), Group D (18.2 ± 6.1), Group E (18.6 ± 6.3)
Gender distribution: 62% female across all groups
Concomitant medications: Varies by group, requiring covariate adjustment

STATISTICAL HYPOTHESIS TESTING ANALYSIS:
This complex clinical trial requires systematic statistical analysis using multiple hypothesis testing procedures, effect size calculations, and sophisticated interpretation of results within regulatory and clinical contexts.

Primary Hypothesis Testing Framework:
Null Hypothesis (H₀): No difference in mean QMG score change between any treatment groups
Alternative Hypothesis (H₁): At least one treatment group differs significantly from others

Global Test Approach:
First, conduct omnibus ANOVA to test overall treatment effect:
F = MSbetween / MSwithin

Observed QMG score changes at 24 months:
Group A: -4.7 ± 8.2 (improvement from baseline 18.4 to 13.7)
Group B: -8.3 ± 7.9 (improvement from baseline 18.1 to 9.8)
Group C: -6.1 ± 8.4 (improvement from baseline 18.7 to 12.6)  
Group D: -9.2 ± 8.1 (improvement from baseline 18.2 to 9.0)
Group E: -7.8 ± 8.6 (improvement from baseline 18.6 to 10.8)

ANOVA Calculation:
Grand mean change: x̄ = (-4.7×580 + -8.3×562 + -6.1×571 + -9.2×567 + -7.8×567) / 2847 = -7.22

Sum of Squares Between Groups (SSB):
SSB = Σnᵢ(x̄ᵢ - x̄)² = 580(-4.7-(-7.22))² + 562(-8.3-(-7.22))² + 571(-6.1-(-7.22))² + 567(-9.2-(-7.22))² + 567(-7.8-(-7.22))²
SSB = 580(2.52)² + 562(-1.08)² + 571(1.12)² + 567(-1.98)² + 567(-0.58)²
SSB = 580(6.35) + 562(1.17) + 571(1.25) + 567(3.92) + 567(0.34)
SSB = 3,683 + 658 + 714 + 2,223 + 193 = 7,471

Mean Square Between (MSB) = SSB / (k-1) = 7,471 / 4 = 1,868

Sum of Squares Within Groups (SSW):
For each group: SSWᵢ = (nᵢ-1) × SDᵢ²
Group A: (580-1) × 8.2² = 579 × 67.24 = 38,932
Group B: (562-1) × 7.9² = 561 × 62.41 = 35,012  
Group C: (571-1) × 8.4² = 570 × 70.56 = 40,219
Group D: (567-1) × 8.1² = 566 × 65.61 = 37,135
Group E: (567-1) × 8.6² = 566 × 73.96 = 41,862
Total SSW = 193,160

Mean Square Within (MSW) = SSW / (N-k) = 193,160 / (2847-5) = 193,160 / 2842 = 67.95

F-statistic = MSB / MSW = 1,868 / 67.95 = 27.49

Critical F-value: F₀.₀₅,₄,₂₈₄₂ ≈ 2.37

Since 27.49 > 2.37, we reject H₀ and conclude significant overall treatment effect (p < 0.001).

Multiple Comparisons Analysis:
With significant omnibus test, proceed to pairwise comparisons using Tukey's HSD adjustment:

HSD = q₀.₀₅,₅,₂₈₄₂ × √(MSW/2 × (1/nᵢ + 1/nⱼ))

For equal sample sizes (~567), q₀.₀₅,₅,₂₈₄₂ ≈ 3.86
HSD ≈ 3.86 × √(67.95/567) = 3.86 × 0.346 = 1.34

Pairwise Comparisons:
|Group D (-9.2) vs Group A (-4.7)| = 4.5 > 1.34 → Significant (p < 0.001)
|Group B (-8.3) vs Group A (-4.7)| = 3.6 > 1.34 → Significant (p < 0.01)  
|Group E (-7.8) vs Group A (-4.7)| = 3.1 > 1.34 → Significant (p < 0.01)
|Group C (-6.1) vs Group A (-4.7)| = 1.4 > 1.34 → Significant (p < 0.05)
|Group D (-9.2) vs Group B (-8.3)| = 0.9 < 1.34 → Not significant
|Group D (-9.2) vs Group C (-6.1)| = 3.1 > 1.34 → Significant (p < 0.01)
|Group D (-9.2) vs Group E (-7.8)| = 1.4 > 1.34 → Significant (p < 0.05)
|Group B (-8.3) vs Group C (-6.1)| = 2.2 > 1.34 → Significant (p < 0.01)
|Group B (-8.3) vs Group E (-7.8)| = 0.5 < 1.34 → Not significant
|Group C (-6.1) vs Group E (-7.8)| = 1.7 > 1.34 → Significant (p < 0.05)

Effect Size Analysis:
Cohen's d for pairwise comparisons using pooled standard deviation:
spooled = √((n₁-1)s₁² + (n₂-1)s₂²)/(n₁+n₂-2))

Group D vs Group A:
spooled = √((566×8.1² + 579×8.2²)/(1145)) = √((37,135 + 38,932)/1145) = √66.44 = 8.15
Cohen's d = (9.2 - 4.7)/8.15 = 0.55 (medium effect size)

Group B vs Group A:  
spooled = √((561×7.9² + 579×8.2²)/(1140)) = √67.82 = 8.23
Cohen's d = (8.3 - 4.7)/8.23 = 0.44 (small to medium effect size)

Analysis of Covariance (ANCOVA):
Adjusting for baseline covariates (age, disease duration, baseline QMG):

ANCOVA Model: Yᵢⱼ = μ + αᵢ + β₁(Age) + β₂(Duration) + β₃(Baseline) + εᵢⱼ

Covariate Effects:
Age: β₁ = -0.08 (older patients show slightly less improvement, p = 0.023)
Disease duration: β₂ = -0.15 (longer duration associated with less improvement, p = 0.007)
Baseline QMG: β₃ = -0.31 (higher baseline scores show greater improvement, p < 0.001)

Adjusted Treatment Effects:
After covariate adjustment:
Group A: -4.9 ± 0.34 (adjusted mean ± SE)
Group B: -8.1 ± 0.34  
Group C: -6.0 ± 0.34
Group D: -9.4 ± 0.34
Group E: -7.7 ± 0.34

ANCOVA F-test: F₄,₂₈₃₉ = 31.2, p < 0.001
R² = 0.086 (8.6% of variance explained by treatment + covariates)

Secondary Endpoint Analysis:
Quality of Life Scale (QoLS) improvements:
Group A: +12.4 ± 18.7, Group B: +28.3 ± 19.2, Group C: +18.7 ± 18.9, Group D: +31.8 ± 19.4, Group E: +25.2 ± 19.1

Anti-VGCC antibody level changes (% reduction):
Group A: -18.2%, Group B: -45.7%, Group C: -32.1%, Group D: -52.3%, Group E: -41.9%

Time-to-event analysis for clinical relapse:
Using Kaplan-Meier survival analysis and Cox proportional hazards:
Median time to relapse: Group A (14.2 months), Group B (22.8 months), Group C (18.4 months), Group D (>24 months), Group E (21.3 months)

Log-rank test: χ² = 47.3, df = 4, p < 0.001
Cox regression hazard ratios (vs Group A):
Group B: HR = 0.58 (95% CI: 0.45-0.74)
Group C: HR = 0.71 (95% CI: 0.56-0.89)  
Group D: HR = 0.39 (95% CI: 0.29-0.53)
Group E: HR = 0.52 (95% CI: 0.41-0.67)

Clinical Significance Assessment:
Minimal clinically important difference (MCID) for QMG score: 3.2 points

Treatment groups exceeding MCID:
Group B: 8.3 > 3.2 ✓ (clinically significant)
Group C: 6.1 > 3.2 ✓ (clinically significant)
Group D: 9.2 > 3.2 ✓ (clinically significant) 
Group E: 7.8 > 3.2 ✓ (clinically significant)
Group A: 4.7 > 3.2 ✓ (clinically significant)

All treatments show clinically meaningful improvements, with Group D (Combination therapy) showing greatest benefit.

Statistical Power and Sample Size Retrospective Analysis:
Achieved power for detecting difference between best (Group D) and worst (Group A) performing treatments:
Effect size = (9.2 - 4.7)/8.15 = 0.55
With n₁ = n₂ ≈ 567, α = 0.05, achieved power > 99%

Therefore, this comprehensive statistical analysis demonstrates that systematic hypothesis testing in complex clinical trials requires careful attention to multiple comparison adjustments, effect size interpretation, covariate analysis, and integration of statistical significance with clinical meaningfulness, ultimately revealing that combination therapy (Group D) provides superior efficacy across multiple endpoints while all treatments exceed clinically important thresholds for patient benefit.
""",
        "params": {
            "max_tokens": 6000,
            "temperature": 0.05,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 36: Multi-Source Intelligence Analysis (Multi-Hop Inference)",
        "prompt": """Document A - Economic Report:
The Baltic Dry Index (BDI) dropped 15% in Q3 2024, indicating reduced global shipping demand. Steel production in China decreased by 8% during the same period. Major shipping routes from Asia to Europe showed 22% fewer cargo vessels. Industrial electricity consumption in Germany fell by 6% in September 2024.

Document B - Corporate Intelligence:
MetalCorp announced layoffs of 3,000 workers in October 2024, citing "market conditions." Their primary operations include steel processing facilities in Ohio and Pennsylvania. The company's stock price fell 28% over Q3 2024. MetalCorp sources 40% of raw materials from Chinese suppliers.

Document C - Financial Markets:
The EUR/USD exchange rate strengthened by 12% in late Q3 2024. European Central Bank maintained interest rates at 4.5%. Manufacturing PMI for the Eurozone dropped to 47.2 in September (below 50 indicates contraction). U.S. Treasury yields increased to 4.8% by end of Q3.

Document D - Supply Chain Report:
Port of Rotterdam reported 18% decline in container throughput for Q3 2024. Average shipping times from Shanghai to Hamburg increased by 5 days. Container rental costs dropped 35% in September 2024. Three major shipping companies reduced their European routes by 30%.

Question: Based on these four documents, construct a comprehensive analysis explaining why MetalCorp's stock price declined and predict the likely impact on their Q4 2024 operations. Your analysis must trace connections across all four documents and identify the causal chains.""",
        "params": {
            "max_tokens": 6500,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 37: Cross-Disciplinary Research Synthesis (Multi-Hop Reasoning)",
        "prompt": """Research Paper A - Neuroscience Study:
"Cognitive Load and Decision Making" (2024): Participants showed 34% slower decision-making when processing more than 7 simultaneous variables. fMRI scans revealed increased prefrontal cortex activation correlating with task complexity. Working memory capacity directly predicted performance on multi-variable problems. Error rates increased exponentially after the 7-variable threshold.

Research Paper B - Computer Science Study:
"Large Language Model Performance on Complex Tasks" (2024): Models with >30B parameters showed diminishing returns on tasks requiring more than 8 sequential logical steps. Attention mechanisms degraded when tracking relationships across more than 12 entities. Chain-of-thought prompting improved performance by 23% on multi-step problems but plateaued after 6 reasoning steps.

Research Paper C - Psychology Study:
"Human-AI Collaboration in Complex Problem Solving" (2024): Teams combining human intuition with AI systematic processing achieved 45% better outcomes than either alone. Humans excelled at identifying relevant variables (first 3-5), while AI performed better at systematic evaluation of all combinations. Optimal handoff point occurred at 6-7 variables.

Research Paper D - Management Study:
"Decision-Making in High-Stakes Business Environments" (2024): Executive teams making decisions with 8+ variables showed 28% more decision paralysis. Companies using structured decision frameworks (limiting to 5-7 key variables) had 31% faster strategic decision implementation. Success rates dropped significantly when analyzing more than 9 simultaneous factors.

Task: Synthesize findings across all four research domains to develop a unified theory explaining the "7±2 rule" in complex cognitive tasks. Connect the neurological basis with computational limitations and practical applications.""",
        "params": {
            "max_tokens": 7000,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 38: Historical Pattern Recognition and Prediction (Multi-Hop Analysis)",
        "prompt": """Historical Document A - Economic Crisis 1929:
Stock market crash October 1929. Bank failures cascaded from rural areas to major cities. Agricultural commodity prices had declined 40% in preceding 18 months. Credit expansion in 1927-1928 led to speculation bubble. Federal Reserve raised interest rates from 3.5% to 6% in 1928-1929. Unemployment rose from 3% to 25% over three years.

Historical Document B - Economic Crisis 1973:
Oil embargo October 1973 increased oil prices 300%. Inflation reached 12% by 1974. Stock market declined 45% from 1973-1974. Nixon wage and price controls created market distortions. Federal Reserve maintained low interest rates despite rising inflation. Unemployment and inflation both exceeded 8% simultaneously.

Historical Document C - Economic Crisis 2008:
Subprime mortgage crisis began 2007. Lehman Brothers collapsed September 2008. Credit markets froze globally. Federal Reserve lowered interest rates to near zero. Government bailouts of major financial institutions totaled $700 billion. Unemployment peaked at 10% in 2009.

Historical Document D - Current Economic Indicators 2024:
Corporate debt levels at historic highs (250% of GDP). Federal Reserve rates at 5.25%, highest in 15 years. Real estate prices up 40% from 2020-2024. Consumer debt service ratios approaching 2007 levels. Bank commercial real estate exposure concentrated in regional banks. Inflation declining but still above target.

Current News Context:
Three regional banks failed in March 2024. Commercial real estate vacancy rates at 25% in major cities. Corporate refinancing cliff approaching in 2024-2025 ($1.4 trillion in debt maturing). Federal deficit projected at $2 trillion for 2024.

Analysis Task: Using pattern recognition from the three historical crises, identify the specific warning indicators present in 2024 that match previous crisis patterns. Trace the causal chains from each historical example and predict the most likely sequence of events for 2024-2025.""",
        "params": {
            "max_tokens": 7500,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 39: Legal Precedent Chain Analysis (Multi-Hop Jurisprudence)",
        "prompt": """Case A - Miranda v. Arizona (1966):
Supreme Court ruled that suspects must be informed of their rights before custodial interrogation. Established the "Miranda Warning" requirement. Reversed Miranda's conviction because confession was obtained without proper warning. Court emphasized Fifth Amendment protection against self-incrimination. Decision applied to all custodial interrogations going forward.

Case B - Dickerson v. United States (2000):
Congressional statute attempted to overturn Miranda by making confession admissibility dependent solely on voluntariness. Supreme Court ruled 7-2 that Miranda was a constitutional decision that Congress cannot overturn by statute. Reaffirmed Miranda as constitutional rule, not just procedural guidance. Chief Justice Rehnquist wrote majority opinion strengthening Miranda.

Case C - Berghuis v. Thompkins (2010):
Suspect remained mostly silent for 2 hours and 45 minutes during interrogation, then made incriminating statement. Court ruled 5-4 that suspects must explicitly invoke right to remain silent. Passive silence does not invoke Miranda protections. Suspect's statement was admissible because he never explicitly invoked his rights.

Case D - Vega v. Tekoh (2022):
Police officer questioned hospital patient without Miranda warning. Patient confessed and information was used in criminal trial. Civil rights lawsuit claimed violation of Miranda rights under Section 1983. Supreme Court ruled 6-3 that Miranda violation alone cannot support federal civil rights claim. Miranda provides trial rights, not basis for civil damages.

Current Case Scenario (2024):
Police arrest suspect for robbery. Suspect is read Miranda rights but responds "I guess I should talk to someone." Officers continue questioning. Suspect provides detailed confession after 45 minutes. Defense argues statement should be suppressed. Prosecution argues suspect never clearly invoked right to counsel.

Legal Question: Trace the evolution of Miranda doctrine through these four cases and predict how the current case would likely be decided. Analyze how each precedent builds on the previous ones and identify the controlling legal principles.""",
        "params": {
            "max_tokens": 6000,
            "temperature": 0.05,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 40: Multi-Domain Scientific Investigation (Cross-Field Inference)",
        "prompt": """Physics Research - Quantum Coherence Study:
Quantum systems maintain coherence for 1.3 milliseconds at 15 millikelvin. Decoherence time decreases exponentially with temperature increases. Electromagnetic interference reduces coherence by 40% even with shielding. System requires isolation from vibrations above 0.01 Hz. Coherence length correlates with material purity (99.99% minimum).

Biology Research - Neural Network Study:
Brain neurons maintain synchronized firing patterns for 1.2 milliseconds during memory consolidation. Neural synchronization breaks down when temperature rises above 37.2°C. Electromagnetic fields (>50 Hz) disrupt neural timing by 35%. Brain requires isolation from external stimuli during deep sleep. Synaptic efficiency correlates with neurotransmitter purity.

Materials Science Research - Superconductor Study:
High-temperature superconductors maintain zero resistance up to 138K. Critical temperature decreases with material impurities. Magnetic fields above 12 Tesla destroy superconducting state. Crystal structure must be isolated from mechanical stress. Superconducting properties correlate with material grain boundaries.

Computer Science Research - Quantum Computing Study:
Quantum processors require error correction every 1.0 milliseconds. Quantum gates fail above 50 millikelvin operating temperature. Classical interference corrupts quantum states above 1 GHz frequency. Quantum systems require vibration isolation below 0.001 Hz. Gate fidelity correlates with fabrication precision (99.9% minimum).

Engineering Context:
New quantum-biological computing system combines quantum coherence with neural network processing. System operates at biological temperatures (37°C) but maintains quantum effects. Preliminary tests show 1.1 millisecond coherence time. Electromagnetic shielding reduces both quantum decoherence and neural disruption.

Research Question: Analyze the convergence patterns across all four domains (physics, biology, materials science, computer science) and explain how the quantum-biological system achieves coherence at biological temperatures when traditional quantum systems require near-absolute zero. Trace the scientific principles from each field that contribute to this breakthrough.""",
        "params": {
            "max_tokens": 8000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 41: Structured Environmental Impact Analysis (Scaffolded Reasoning Template)",
        "prompt": """SCENARIO: A proposed lithium mining operation in Northern Chile's Atacama Desert will extract 50,000 tons of lithium annually over 20 years. The site is located 15km from a flamingo breeding ground and 8km from indigenous Atacameño communities who depend on limited groundwater resources.

Your task is to conduct a comprehensive environmental impact analysis using the following MANDATORY reasoning structure:

**ANALYSIS SECTION:**
Begin by systematically identifying all stakeholder groups, environmental systems, and potential impact vectors. Consider direct, indirect, and cumulative effects across temporal and spatial scales.

**EVIDENCE SECTION:**  
Present specific data points, scientific studies, regulatory standards, and precedent cases that inform your analysis. Include quantitative metrics where possible (water usage rates, contamination thresholds, biodiversity indices, etc.).

**REASONING SECTION:**
Connect your evidence to logical conclusions through explicit causal chains. Show how each piece of evidence supports or contradicts specific impact predictions. Address uncertainty and alternative interpretations.

**CONCLUSION SECTION:**
Synthesize your analysis into clear recommendations with risk assessments. Prioritize impacts by severity, likelihood, and reversibility. Propose specific mitigation measures.

**VERIFICATION SECTION:**
Challenge your own conclusions by identifying potential blind spots, questioning key assumptions, and suggesting additional data that would strengthen or weaken your analysis. Consider how opposing stakeholders might critique your reasoning.

CRITICAL REQUIREMENT: You must explicitly label each section and demonstrate how your reasoning flows logically from evidence to conclusions. Show your work at each step.""",
        "params": {
            "max_tokens": 8000,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 42: Structured Medical Diagnosis Protocol (Reasoning Scaffolding)",
        "prompt": """CASE PRESENTATION: 
67-year-old male presents to emergency department with:
- Acute onset chest pain (8/10 severity) radiating to left arm
- Shortness of breath, diaphoresis, nausea  
- History: Type 2 diabetes (15 years), hypertension, smoking (40 pack-years, quit 5 years ago)
- Medications: Metformin, Lisinopril, Aspirin
- Vitals: BP 180/110, HR 105, RR 22, O2 sat 94% on room air
- Initial ECG shows ST elevation in leads II, III, aVF

Complete your diagnostic reasoning using this MANDATORY structured approach:

**ANALYSIS SECTION:**
Systematically review the presentation, identifying all relevant clinical features, risk factors, and potential diagnostic categories. Consider the differential diagnosis framework: common, can't miss, and atypical presentations.

**EVIDENCE SECTION:**  
Present specific clinical criteria, laboratory values, imaging findings, and evidence-based diagnostic algorithms that apply to this case. Reference validated clinical decision rules and diagnostic accuracy statistics.

**REASONING SECTION:**
Apply diagnostic reasoning principles: pattern recognition, probabilistic thinking, and hypothesis testing. Show how you weight different findings, account for pre-test probability, and integrate multiple sources of evidence. Address diagnostic uncertainty.

**CONCLUSION SECTION:**
State your primary diagnosis with confidence level, list key differential diagnoses with probability estimates, and outline immediate management priorities with time-sensitive interventions.

**VERIFICATION SECTION:**
Critically evaluate your diagnostic reasoning by identifying potential cognitive biases, alternative interpretations of findings, and additional tests that would confirm or refute your diagnosis. Consider what you might have missed and how the case could evolve.

Each section must be clearly labeled and show explicit logical connections between clinical evidence and diagnostic conclusions.""",
        "params": {
            "max_tokens": 7000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 43: Structured Investment Analysis Framework (Systematic Reasoning)",
        "prompt": """INVESTMENT SCENARIO:
Quantum Computing Dynamics (QCD) is a 5-year-old quantum computing startup seeking $200M Series C funding. Key details:
- Technology: 64-qubit fault-tolerant quantum processor  
- Competition: IBM (127-qubit system), Google (70-qubit system), Rigetti (32-qubit system)
- Market: Quantum computing market projected $65B by 2030 (currently $1.3B)
- Financials: $45M revenue (2023), 300% growth rate, $180M total funding raised
- Team: Former IBM/Google quantum researchers, 150 employees
- Partnerships: Microsoft Azure integration, pilot programs with JPMorgan, Roche

Your task: Provide a comprehensive investment recommendation using the structured reasoning framework:

**ANALYSIS SECTION:**
Systematically evaluate the investment across multiple dimensions: technology differentiation, market opportunity, competitive positioning, team quality, financial performance, and risk factors. Define your evaluation criteria and methodology.

**EVIDENCE SECTION:**  
Present quantitative metrics, market data, technical specifications, financial ratios, and comparable company analyses that inform your assessment. Include industry benchmarks, expert opinions, and third-party validations.

**REASONING SECTION:**
Connect your evidence to investment conclusions through explicit valuation logic. Show how you weight different factors, account for various scenarios (bull/base/bear cases), and handle uncertainty in emerging technology markets. Address key assumptions and their sensitivity.

**CONCLUSION SECTION:**
State your investment recommendation (invest/pass) with conviction level, proposed valuation and terms, and key milestones for monitoring. Include position sizing and portfolio construction considerations.

**VERIFICATION SECTION:**
Stress-test your investment thesis by identifying the strongest counter-arguments, questioning optimistic projections, and considering how the investment could fail. Challenge your assumptions and biases. What evidence would change your recommendation?

Demonstrate rigorous analytical thinking with clear logical progression from data to recommendation.""",
        "params": {
            "max_tokens": 7500,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 44: Structured Policy Analysis Template (Government Decision Framework)",
        "prompt": """POLICY CHALLENGE:
The European Union is considering a comprehensive ban on single-use plastics by 2027, including food packaging, coffee cups, cutlery, and shopping bags. This follows existing restrictions on specific items but would expand to cover 80% of plastic waste. 

Affected stakeholders:
- 30 million EU jobs in plastics industry ($400B annual revenue)
- 450 million consumers across 27 member states
- Environmental groups citing 25 million tons annual plastic waste
- Restaurant/retail industries requiring alternative packaging
- Waste management companies ($50B recycling industry)

Your task: Develop a comprehensive policy recommendation using mandatory structured reasoning:

**ANALYSIS SECTION:**
Map all stakeholder interests, policy objectives, and implementation challenges. Consider economic impacts, environmental benefits, technological alternatives, enforcement mechanisms, and international trade implications. Define success metrics.

**EVIDENCE SECTION:**  
Present empirical data on plastic pollution impacts, economic costs/benefits, alternative material performance, successful policy precedents (other jurisdictions), industry transition timelines, and consumer behavior research. Include cost-benefit calculations.

**REASONING SECTION:**
Apply policy analysis frameworks: feasibility assessment, stakeholder impact evaluation, unintended consequence prediction, and implementation pathway design. Show how you balance competing objectives and handle trade-offs between economic and environmental goals.

**CONCLUSION SECTION:**
Recommend specific policy design with implementation timeline, stakeholder transition support, enforcement mechanisms, and performance monitoring. Address EU member state coordination and competitiveness concerns.

**VERIFICATION SECTION:**
Critically examine your policy recommendation by modeling opposition arguments, identifying implementation bottlenecks, and considering how the policy could backfire. What assumptions are most vulnerable? How would different interest groups attack your proposal?

Structure your analysis to show clear logical flow from stakeholder analysis through evidence evaluation to policy design and risk assessment.""",
        "params": {
            "max_tokens": 8500,
            "temperature": 0.25,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 45: Structured Technology Assessment Protocol (Innovation Analysis Framework)",
        "prompt": """TECHNOLOGY ASSESSMENT SCENARIO:
Brain-Computer Interfaces (BCIs) are approaching clinical viability for treating paralysis, depression, and cognitive enhancement. Neuralink, Synchron, and other companies are conducting human trials with implantable neural chips.

Current status:
- 4 patients with Neuralink implants controlling computers/phones through thought
- FDA approval for limited medical applications (paralysis, epilepsy)
- Technical challenges: signal degradation, immune response, battery life
- Ethical concerns: privacy, autonomy, enhancement vs. treatment
- Economic potential: $40B market by 2030, transformative for disability care
- Regulatory uncertainty across US, EU, and other jurisdictions

Your mission: Complete a comprehensive technology assessment using the structured reasoning framework:

**ANALYSIS SECTION:**
Define the technology landscape, identify key applications (medical/enhancement/consumer), map stakeholder ecosystems (patients, physicians, regulators, tech companies, ethicists), and establish evaluation criteria across technical, ethical, economic, and social dimensions.

**EVIDENCE SECTION:**  
Compile technical performance data, clinical trial results, regulatory precedents, ethical analysis frameworks, economic projections, public opinion surveys, and expert consensus positions. Include safety profiles, efficacy measurements, and cost-effectiveness studies.

**REASONING SECTION:**
Apply technology assessment methodologies: risk-benefit analysis, ethical principle evaluation, innovation diffusion modeling, and regulatory pathway prediction. Show how you integrate technical feasibility with social acceptability and address uncertainty in emerging technology predictions.

**CONCLUSION SECTION:**
Provide specific recommendations for technology development priorities, regulatory framework design, ethical safeguards, and societal preparation strategies. Include timeline predictions and milestone markers for decision points.

**VERIFICATION SECTION:**
Challenge your technology assessment by considering alternative scenarios, questioning expert predictions, and identifying blind spots in current analysis. How might your assessment be wrong? What developments could dramatically alter the trajectory?

Your analysis must demonstrate sophisticated reasoning about complex socio-technical systems with explicit logical connections between evidence and conclusions.""",
        "params": {
            "max_tokens": 9000,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 46: Reverse Engineering Crime Scene Analysis (Backward Reasoning)",
        "prompt": """CONCLUSION GIVEN: Detective Sarah Martinez concluded that the robbery at First National Bank was an inside job orchestrated by someone with intimate knowledge of the security systems and staff schedules.

Your task: Work BACKWARD from this conclusion to identify what evidence and logical reasoning steps would support this determination. You must reverse-engineer the investigative process.

BACKWARD REASONING REQUIREMENTS:

1. **Evidence Identification**: What specific pieces of evidence would lead to this conclusion? Consider physical evidence, behavioral indicators, timing patterns, technical knowledge requirements, and witness testimonies.

2. **Logical Chain Reconstruction**: Trace the reasoning path from raw evidence to intermediate conclusions to final determination. Show how each piece of evidence eliminates alternative theories.

3. **Alternative Theory Elimination**: Demonstrate how this conclusion rules out external robbery scenarios. What evidence would contradict "random external robbery" or "professional heist team" theories?

4. **Knowledge Requirements Analysis**: Work backwards to identify what insider information would be necessary to execute this crime. What systems knowledge, schedule awareness, and procedural understanding does the evidence suggest?

5. **Suspect Pool Narrowing**: Based on your evidence requirements, trace back to identify the characteristics and access levels the perpetrator must have possessed.

Challenge: You must construct a coherent investigative narrative working from conclusion to evidence, showing how Detective Martinez would have arrived at her determination through systematic elimination of alternatives.""",
        "params": {
            "max_tokens": 7000,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 47: Reverse Medical Diagnosis Reasoning (Conclusion-to-Evidence Tracing)",
        "prompt": """MEDICAL CONCLUSION: Dr. Chen diagnosed the patient with Addison's disease (primary adrenal insufficiency) rather than depression, chronic fatigue syndrome, or other conditions with similar presentations.

PATIENT CONTEXT: 34-year-old female presenting with 6-month history of fatigue, weight loss, mood changes, and intermittent nausea.

Your mission: Work BACKWARD from this specific diagnosis to reconstruct the clinical reasoning process that led to this conclusion.

REVERSE DIAGNOSTIC REASONING:

1. **Diagnostic Criteria Reconstruction**: What specific clinical criteria, laboratory values, and examination findings would be necessary to support this diagnosis over alternatives? Work backwards from diagnostic certainty to required evidence.

2. **Differential Diagnosis Elimination**: Trace the reasoning that eliminated depression, chronic fatigue syndrome, thyroid disorders, and other conditions. What evidence patterns distinguish Addison's disease from these alternatives?

3. **Clinical Clue Integration**: Identify the subtle clinical signs and symptoms that, when combined, point specifically toward adrenal insufficiency. How would these findings create a diagnostic pattern?

4. **Testing Strategy Reconstruction**: What sequence of diagnostic tests would logically lead to this conclusion? Work backwards from diagnosis to identify the critical test results that confirmed the suspicion.

5. **Timeline Analysis**: Reconstruct how the 6-month symptom progression would match Addison's disease pathophysiology versus other conditions.

Demonstrate how multiple seemingly non-specific symptoms combine to create a diagnostic signature that points specifically toward primary adrenal insufficiency.""",
        "params": {
            "max_tokens": 6500,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 48: Reverse Engineering Market Crash Analysis (Financial Backward Reasoning)",
        "prompt": """MARKET CONCLUSION: Financial analysts concluded that the September 2024 tech stock crash was primarily driven by institutional algorithmic trading amplifying an initial liquidity crisis, rather than fundamental valuation concerns or retail panic selling.

CRASH CONTEXT: 
- NASDAQ fell 12% in 3 days
- Tech stocks disproportionately affected (FAANG down 15-20%)
- Volume exceeded normal by 400%
- Volatility index spiked to 35
- Recovery began after institutional trading hours

Your task: Work BACKWARD from this specific conclusion to identify the evidence patterns and reasoning chain that would support this analysis.

REVERSE FINANCIAL ANALYSIS:

1. **Evidence Pattern Recognition**: What specific market data signatures would indicate algorithmic amplification versus fundamental selling? Trace backwards from conclusion to required data patterns.

2. **Alternative Theory Elimination**: How would the evidence rule out "fundamental repricing," "retail panic," or "coordinated manipulation" theories? Work backwards to show what evidence contradicts these alternatives.

3. **Algorithmic Trading Indicators**: Identify the specific trading patterns, timing signatures, and volume characteristics that would point to algorithmic amplification of an initial trigger event.

4. **Liquidity Crisis Identification**: What market microstructure evidence would indicate the initial trigger was liquidity-based rather than news-driven or sentiment-driven?

5. **Institutional vs Retail Analysis**: Reconstruct the trading pattern analysis that would distinguish institutional algorithmic behavior from retail panic selling during the crash period.

Demonstrate how market data forensics can work backwards from a crash explanation to identify the specific evidence signatures that support this particular theory over alternatives.""",
        "params": {
            "max_tokens": 7500,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 49: Reverse Archaeological Interpretation (Cultural Conclusion Tracing)",
        "prompt": """ARCHAEOLOGICAL CONCLUSION: The excavation team concluded that the Neolithic settlement at site XJ-47 was abandoned suddenly due to environmental catastrophe (likely volcanic activity) rather than warfare, disease, or gradual migration.

SITE CONTEXT:
- 5,000-year-old settlement with 200+ structures
- Artifacts left in place suggesting rapid departure
- No evidence of defensive preparations or conflict
- Settlement never reoccupied despite favorable location
- Regional geological activity documented in nearby sites

Your challenge: Work BACKWARD from this environmental catastrophe conclusion to reconstruct the archaeological evidence and reasoning process.

REVERSE ARCHAEOLOGICAL REASONING:

1. **Artifact Pattern Analysis**: What specific patterns in artifact distribution, preservation, and abandonment would support sudden environmental departure versus other theories? Trace back from conclusion to required material evidence.

2. **Stratigraphy Reconstruction**: Identify the geological and stratigraphic evidence layers that would indicate environmental catastrophe timing and impact on the settlement.

3. **Alternative Theory Elimination**: How would the archaeological evidence rule out warfare (no weapons, fortifications, skeletal trauma), disease (no mass burials, ritual deposits), or planned migration (organized clearing, selective artifact removal)?

4. **Environmental Impact Indicators**: Work backwards to identify what environmental signatures (volcanic ash layers, climate proxies, ecological disruption evidence) would support the catastrophic environmental change theory.

5. **Regional Context Integration**: Reconstruct how regional archaeological and geological data would corroborate local site evidence for environmental disaster affecting multiple settlements simultaneously.

Show how multiple lines of archaeological evidence combine to create a compelling case for environmental catastrophe as the abandonment cause.""",
        "params": {
            "max_tokens": 8000,
            "temperature": 0.25,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Complex Test 50: Reverse Engineering Product Failure Analysis (Technical Backward Reasoning)",
        "prompt": """ENGINEERING CONCLUSION: The investigation team determined that the autonomous vehicle accident was caused by a cascading sensor fusion failure triggered by specific environmental conditions (heavy rain + construction zone reflective surfaces) that created false positive obstacles, leading to emergency braking on a highway.

ACCIDENT CONTEXT:
- Autonomous vehicle engaged emergency braking at 65 mph
- No actual obstacles present in vehicle path
- Multiple vehicles involved in resulting collision
- Weather: heavy rain, reduced visibility
- Location: Active construction zone with temporary barriers

Your mission: Work BACKWARD from this specific technical conclusion to reconstruct the engineering analysis that identified this failure mode.

REVERSE ENGINEERING ANALYSIS:

1. **Sensor Data Reconstruction**: What specific patterns in lidar, camera, and radar data would indicate false positive detection versus genuine obstacle avoidance? Trace backwards from conclusion to required sensor signatures.

2. **Environmental Factor Isolation**: How would the analysis isolate the contribution of rain and reflective construction surfaces to the sensor fusion failure? Work backwards to identify the environmental testing that would validate this theory.

3. **Software Logic Tracing**: Reconstruct the decision tree analysis that would show how false sensor inputs propagated through the autonomous driving algorithms to trigger emergency braking.

4. **Alternative Failure Mode Elimination**: How would the evidence rule out mechanical failure, cyberattack, software bug, or driver error? Work backwards to show what data contradicts these alternative theories.

5. **System Integration Analysis**: Identify how the investigation would trace the failure from environmental conditions through sensor processing to fusion algorithms to vehicle control systems.

Demonstrate how technical forensics can work backwards from a complex system failure to identify the specific environmental trigger conditions and cascading failure modes.""",
        "params": {
            "max_tokens": 8500,
            "temperature": 0.15,
            "top_p": 0.95,
            "stream": False
        }
    }
]

def run_performance_test(test_case):
    """Sends a request to the base model API and logs the results."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    print("--- Prompt Preview ---")
    print(textwrap.shorten(prompt, width=300, placeholder="..."))
    print("-" * 40)

    payload = {
        "model": "your-base-model-name",  # <-- IMPORTANT: Change this to your model's name
        "prompt": prompt,
        **params
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=600) # Increased timeout for very long context
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        total_duration = end_time - start_time
        
        completion_text = response_data.get("choices", [{}])[0].get("text", "")
        
        # Extract token usage for performance metrics
        completion_tokens = 0
        prompt_tokens = 0
        if "usage" in response_data:
            completion_tokens = response_data["usage"].get("completion_tokens", 0)
            prompt_tokens = response_data["usage"].get("prompt_tokens", 0)
        
        tokens_per_second = 0
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration

        # Save Response
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_completion.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{'-'*20}\n{prompt}\n\n")
            f.write(f"COMPLETION:\n{'-'*20}\n{completion_text}\n\n")
            f.write(f"METRICS:\n{'-'*20}\n")
            f.write(f"Duration: {total_duration:.2f}s\n")
            f.write(f"Prompt Tokens: {prompt_tokens}\n")
            f.write(f"Completion Tokens: {completion_tokens}\n")
            f.write(f"Tokens per Second: {tokens_per_second:.2f} T/s\n")
                
        print(f"--- ✅ SUCCESS: Request completed in {total_duration:.2f} seconds ---")
        print(f"--- 📊 Performance: {tokens_per_second:.2f} T/s ---")
        print(f"--- 📝 Tokens Generated: {completion_tokens} ---")
        print(f"--- Full completion saved to '{output_path}' ---")
        return True

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timed out for '{name}'")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Request failed for '{name}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during '{name}': {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED LONG-CONTEXT REASONING TEST SUITE")
    print(f"Server: {API_URL}")
    print(f"Total Tests: {len(COMPLEX_TEST_CASES)}")
    print("=" * 80)
    
    successful_tests = 0
    
    for i, test in enumerate(COMPLEX_TEST_CASES):
        print(f"\n\n[{i+1}/{len(COMPLEX_TEST_CASES)}] Preparing test...")
        if run_performance_test(test):
            successful_tests += 1
        time.sleep(1)
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Successful Tests: {successful_tests}/{len(COMPLEX_TEST_CASES)}")
    print(f"Success Rate: {(successful_tests/len(COMPLEX_TEST_CASES)*100):.1f}%")
