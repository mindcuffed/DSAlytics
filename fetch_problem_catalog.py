import requests
import json
import time
import os


# ============================================================
# DSAlytics - LeetCode Problem Catalog Collector
# ============================================================

GRAPHQL_URL = "https://leetcode.com/graphql"

OUTPUT_FILE = "data/raw/leetcode_problems.json"

BATCH_SIZE = 100

HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com/problemset/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}


# ============================================================
# GRAPHQL QUERY
# ============================================================

QUERY = """
query problemsetQuestionListV2(
    $skip: Int
    $limit: Int
    $filters: QuestionFilterInput
) {
    problemsetQuestionListV2(
        skip: $skip
        limit: $limit
        filters: $filters
    ) {
        questions {
            questionFrontendId
            title
            titleSlug
            difficulty
            paidOnly
            topicTags {
                name
                slug
            }
        }
    }
}
"""


# ============================================================
# FETCH ONE BATCH
# ============================================================

def fetch_problems(skip, limit):

    variables = {
        "skip": skip,
        "limit": limit,
        "filters": {
            "filterCombineType": "ALL"
        }
    }

    response = requests.post(
        GRAPHQL_URL,
        json={
            "query": QUERY,
            "variables": variables
        },
        headers=HEADERS,
        timeout=30
    )

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:

        print("\nLeetCode response:")
        print(response.text[:3000])

        return None

    data = response.json()

    if "errors" in data:

        print("\nGraphQL errors:")

        print(
            json.dumps(
                data["errors"],
                indent=2
            )
        )

        return None

    return data["data"]["problemsetQuestionListV2"]["questions"]


# ============================================================
# FORMAT PROBLEM
# ============================================================

def format_problem(question):

    return {
        "frontend_id": question["questionFrontendId"],
        "title": question["title"],
        "titleSlug": question["titleSlug"],
        "difficulty": question["difficulty"],
        "topics": [
            tag["name"]
            for tag in question["topicTags"]
        ],
        "isPaidOnly": question["paidOnly"]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DSAlytics - LeetCode Problem Catalog Collector")
    print("=" * 60)

    print("\nConnecting to LeetCode...")

    all_problems = []

    skip = 0

    # ========================================================
    # FETCH BATCHES
    # ========================================================

    while True:

        print(
            f"\nFetching problems "
            f"{skip + 1} - "
            f"{skip + BATCH_SIZE}..."
        )

        questions = fetch_problems(
            skip,
            BATCH_SIZE
        )

        if questions is None:

            print(
                "\nCould not retrieve this batch."
            )

            # If nothing has been collected,
            # do NOT overwrite the catalog.

            if len(all_problems) == 0:

                print(
                    "\nNo data collected."
                )

                print(
                    "Existing catalog was not modified."
                )

                return

            break

        if not questions:

            print(
                "\nNo more problems returned."
            )

            break

        for question in questions:

            all_problems.append(
                format_problem(question)
            )

        print(
            f"Received {len(questions)} problems."
        )

        print(
            f"Total collected: "
            f"{len(all_problems)}"
        )

        # Last batch

        if len(questions) < BATCH_SIZE:

            print(
                "\nReached the end of the catalog."
            )

            break

        skip += BATCH_SIZE

        time.sleep(0.5)

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_problems = {}

    for problem in all_problems:

        unique_problems[
            problem["titleSlug"]
        ] = problem

    all_problems = list(
        unique_problems.values()
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_problems,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    easy = sum(
        1
        for p in all_problems
        if p["difficulty"] == "EASY"
    )

    medium = sum(
        1
        for p in all_problems
        if p["difficulty"] == "MEDIUM"
    )

    hard = sum(
        1
        for p in all_problems
        if p["difficulty"] == "HARD"
    )

    premium = sum(
        1
        for p in all_problems
        if p["isPaidOnly"]
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"\nUnique problems collected: "
        f"{len(all_problems)}"
    )

    print(
        f"Easy   : {easy}"
    )

    print(
        f"Medium : {medium}"
    )

    print(
        f"Hard   : {hard}"
    )

    print(
        f"Premium: {premium}"
    )

    print(
        "\nFile created:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )

    print(
        "\nData collection complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()