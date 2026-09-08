import pandas as pd


def load_questions(questions_csv: str) -> pd.DataFrame:
    df = pd.read_csv(questions_csv)
    print(f"✅ Questions chargées : {len(df)}")
    return df


def load_article_ids(articles_csv: str) -> set:
    df = pd.read_csv(articles_csv)
    ids = set(df["article_id"].dropna().astype(str).unique())
    print(f"✅ Articles chargés : {len(ids)}")
    return ids


def filter_questions_by_articles(df_questions: pd.DataFrame, article_ids: set) -> pd.DataFrame:
    out = df_questions[df_questions["article_id"].astype(str).isin(article_ids)].copy()
    print(f"✅ Questions filtrées : {len(out)}")
    return out


def sample_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
