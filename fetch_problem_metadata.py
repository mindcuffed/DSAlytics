import requests
import json
import csv
import time
import os

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

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
    "Referer": "https://leetcode.com/",
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
# GRAPHQL QUERY
# ============================================================

QUESTION_QUERY = """
query questionData($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
        questionId
        questionFrontendId
        title
        titleSlug
        difficulty
        topicTags {
            name
            slug
        }
    }
}
"""


# ============================================================
# RUN GRAPHQL QUERY
# ============================================================

def get_problem_metadata(title_slug):

    response = requests.post(
        GRAPHQL_URL,
        json={
            "query": QUESTION_QUERY,
            "variables": {
                "titleSlug": title_slug
            }
        },
        headers=HEADERS,
        cookies=COOKIES,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if "errors" in result:
        print(
            f"GraphQL error for {title_slug}: "
            f"{result['errors']}"
        )
        return None

    return result.get("data", {}).get("question")


# ============================================================
# LOAD SUBMISSIONS
# ============================================================

def load_submissions():

    filename = "data/raw/leetcode_submissions.csv"

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        submissions = list(reader)

    return submissions


# ============================================================
# GET UNIQUE PROBLEMS
# ============================================================

def get_unique_problems(submissions):

    problems = {}

    for submission in submissions:

        slug = submission["titleSlug"]

        if slug not in problems:

            problems[slug] = submission["title"]

    return problems


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(metadata):

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    json_file = "data/raw/problem_metadata.json"

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_file = "data/raw/problem_metadata.csv"

    fields = [
        "questionId",
        "questionFrontendId",
        "title",
        "titleSlug",
        "difficulty",
        "topics"
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for problem in metadata:

            writer.writerow({
                "questionId": problem.get("questionId"),
                "questionFrontendId": problem.get(
                    "questionFrontendId"
                ),
                "title": problem.get("title"),
                "titleSlug": problem.get("titleSlug"),
                "difficulty": problem.get("difficulty"),
                "topics": "|".join(
                    problem.get("topics", [])
                )
            })

    return json_file, csv_file


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DSAlytics - Problem Metadata Collector")
    print("=" * 60)

    print("\nLoading submission data...")

    submissions = load_submissions()

    print(
        f"Loaded {len(submissions)} submissions."
    )

    problems = get_unique_problems(
        submissions
    )

    print(
        f"Found {len(problems)} unique problems."
    )

    print("\nFetching problem metadata...\n")

    metadata = []

    for index, (slug, title) in enumerate(
        problems.items(),
        start=1
    ):

        print(
            f"[{index}/{len(problems)}] "
            f"{title}"
        )

        problem = get_problem_metadata(slug)

        if problem:

            metadata.append({
                "questionId": problem.get(
                    "questionId"
                ),
                "questionFrontendId": problem.get(
                    "questionFrontendId"
                ),
                "title": problem.get(
                    "title"
                ),
                "titleSlug": problem.get(
                    "titleSlug"
                ),
                "difficulty": problem.get(
                    "difficulty"
                ),
                "topics": [
                    tag["name"]
                    for tag in problem.get(
                        "topicTags",
                        []
                    )
                ]
            })

        else:

            print(
                f"Could not fetch metadata for {title}"
            )

        # Avoid hammering LeetCode
        time.sleep(0.5)

    json_file, csv_file = save_metadata(
        metadata
    )

    print("\n" + "=" * 60)
    print(
        f"METADATA COLLECTED: {len(metadata)}"
    )
    print("=" * 60)

    print("\nFiles created:")

    print(f"✓ {json_file}")
    print(f"✓ {csv_file}")

    print("\nMetadata collection complete.")