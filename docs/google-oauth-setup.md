# Google Calendar OAuth Setup (Local Development)

Follow these steps once to let the agent talk to your Google Calendar:

1. **Create or select a Google Cloud project**
   - Open [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (e.g. `calendar-agent-dev`).

2. **Enable the Google Calendar API**
   - Go to **APIs & Services → Library**.
   - Search for **Google Calendar API** and click **Enable**.

3. **Configure the OAuth consent screen**
   - Go to **Google Auth Platform → Audience** (or **APIs & Services → OAuth consent screen**).
   - Choose **External** (for personal Gmail accounts) and keep the publishing status as **Testing**.
   - Under **Test users**, add your own Google email address.
   - Under **Data Access / Scopes**, add `https://www.googleapis.com/auth/calendar.events`.

4. **Create Desktop OAuth credentials**
   - Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
   - Application type: **Desktop app**.
   - Download the JSON file and save it as `credentials.json` in the repo root.

5. **First login**
   - Both `credentials.json` and `token.json` are listed in `.gitignore` so they will never be committed.
   - The first time you run `GoogleCalendarClient.from_oauth_files()`, a browser window will open to approve calendar access and cache a refreshable `token.json` locally.
