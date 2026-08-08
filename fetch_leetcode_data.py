import requests
import json
import time
import os
import csv

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

USERNAME = "FcVqry3N6T"

LEETCODE_SESSION = os.getenv("LEETCODE_SESSION")
CSRF_TOKEN = os.getenv("CSRF_TOKEN")

if not LEETCODE_SESSION or not CSRF_TOKEN:
    raise ValueError(
        "LEETCODE_SESSION or CSRF_TOKEN is missing from .env"
    )


# ============================================================
# LEETCODE CONFIGURATION
# ============================================================

GRAPHQL_URL = "https://leetcode.com/graphql"

HEADERS = {
    "Content-Type": "application/json",
    "Referer": f"https://leetcode.com/{USERNAME}/",
    "Origin": "https://leetcode.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "x-csrftoken": CSRF_TOKEN
}

COOKIES = {
    "LEETCODE_SESSION": LEETCODE_SESSION,
    "csrftoken": CSRF_TOKEN
}


# ============================================================
# GRAPHQL REQUEST
# ============================================================

def run_query(query, variables):

    response = requests.post(
        GRAPHQL_URL,
        json={
            "query": query,
            "variables": variables
        },
        headers=HEADERS,
        cookies=COOKIES,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if "errors" in result:
        raise RuntimeError(
            f"LeetCode GraphQL error: {result['errors']}"
        )

    return result


# ============================================================
# SUBMISSION HISTORY QUERY
# ============================================================

SUBMISSION_QUERY = """
query submissionList($offset: Int!, $limit: Int!) {
    submissionList(
        offset: $offset
        limit: $limit
    ) {
        lastKey
        hasNext

        submissions {
            id
            title
            titleSlug
            statusDisplay
            lang
            runtime
            memory
            timestamp
        }
    }
}
"""


# ============================================================
# GET ONE PAGE
# ============================================================

def get_submissions(offset=0, limit=20):

    return run_query(
        SUBMISSION_QUERY,
        {
            "offset": offset,
            "limit": limit
        }
    )


# ============================================================
# GET ALL SUBMISSIONS
# ============================================================

def get_all_submissions():

    all_submissions = []

    offset = 0
    limit = 20

    while True:

        print(f"Fetching submissions {offset + 1} - {offset + limit}...")

        result = get_submissions(
            offset=offset,
            limit=limit
        )

        submission_data = result["data"]["submissionList"]

        submissions = submission_data["submissions"]

        if not submissions:
            print("No more submissions returned.")
            break

        all_submissions.extend(submissions)

        print(
            f"Received {len(submissions)} submissions. "
            f"Total: {len(all_submissions)}"
        )

        if not submission_data["hasNext"]:
            print("Reached the end of submission history.")
            break

        offset += limit

        # Avoid hitting LeetCode too quickly
        time.sleep(1)

    return all_submissions


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(submissions, filename):

    if not submissions:
        print("No submissions to save.")
        return

    fields = [
        "id",
        "title",
        "titleSlug",
        "statusDisplay",
        "lang",
        "runtime",
        "memory",
        "timestamp"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for submission in submissions:

            writer.writerow({
                field: submission.get(field)
                for field in fields
            })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DSAlytics - LeetCode Data Collector")
    print("=" * 60)

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    print("\nStarting submission collection...\n")

    submissions = get_all_submissions()

    print("\n" + "=" * 60)
    print(f"TOTAL SUBMISSIONS COLLECTED: {len(submissions)}")
    print("=" * 60)

    # Save JSON
    json_file = "data/raw/leetcode_submissions.json"

    save_json(
        submissions,
        json_file
    )

    # Save CSV
    csv_file = "data/raw/leetcode_submissions.csv"

    save_csv(
        submissions,
        csv_file
    )

    print("\nFiles created:")

    print(f"✓ {json_file}")
    print(f"✓ {csv_file}")

    print("\nData collection complete.")