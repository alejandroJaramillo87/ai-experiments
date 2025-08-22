import requests
import json
import time
import os
import textwrap

# Configuration for base model completions
API_URL = "http://127.0.0.1:8004/v1/completions"  # Completions endpoint, not chat
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
