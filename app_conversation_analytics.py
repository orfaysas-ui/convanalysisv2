import streamlit as st
import pandas as pd
import io
import utils


st.title("Analyse automatique des conversations Butler")


# =============================================================================
# Lecture des fichiers uploadés
# =============================================================================

def read_file(file) -> pd.DataFrame:
    """Lit un fichier Excel ou CSV uploadé via Streamlit."""
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    if file.name.endswith(".csv"):
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=encoding)
            except Exception:
                pass
        raise ValueError(f"Impossible de lire {file.name} avec les encodages testés.")

    raise ValueError(f"Format non supporté : {file.name}")


def parse_started_date(df: pd.DataFrame) -> pd.Series:
    """Parse la colonne 'started' en datetime avec timezone Europe/Paris."""
    return pd.to_datetime(
        df["started"].str.replace(" Europe/Paris", "", regex=False),
        errors="coerce",
    ).dt.tz_localize("Europe/Paris")


# =============================================================================
# Construction du transcript lisible
# =============================================================================

SPEAKER_LABELS = {
    "user": "USER",
    "bot": "BOT",
    "human_agent": "AGENT",
    "csat": "CSAT",
    "other": "OTHER",
    "exclude": "EXCLUDE",
}


def build_readable_transcript(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un transcript lisible par conversation, formaté comme :
    [YYYY-MM-DD HH:MM:SS] LOCUTEUR : message

    Retourne un DataFrame avec une colonne 'transcript_lisible' par id.
    """
    df = questions.copy()
    df["speaker_label"] = df["speaker_type"].map(SPEAKER_LABELS).fillna("OTHER")
    df["line"] = (
        "["
        + df["heure_message_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "] "
        + df["speaker_label"]
        + " : "
        + df["message"].fillna("").astype(str)
    )

    return (
        df.groupby("id")["line"]
        .apply(lambda x: "\n".join(x))
        .reset_index()
        .rename(columns={"line": "transcript_lisible"})
    )


# =============================================================================
# Contenu de la question (texte + datetime)
# =============================================================================

def get_question_content(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait le texte et la date de la première vraie question utilisateur
    pour chaque (id, question_id).

    Utilise 'is_new_question_start' comme critère principal,
    avec fallback sur le premier message user de la question.
    """
    df = questions.copy()

    is_user = df["speaker_type"] == "user"
    has_question = df["question_id"].notna()
    # On exclut les messages qui ne sont que des pièces jointes ou liens
    not_attachment = ~df["message"].fillna("").str.contains(utils.PATTERN_ATTACHMENT_OR_LINK, na=False)

    # Cible principale : premier message marqué comme début de question (hors attachment)
    primary = (
        df[is_user & has_question & df["is_new_question_start"] & not_attachment]
        .sort_values(["id", "question_id", "ordre_message_question"])
        .groupby(["id", "question_id"])
        .agg(
            datetime_question=("heure_message_dt", "first"),
            question=("message", "first"),
        )
        .reset_index()
    )

    # Fallback 1 : premier message user non-attachment de la question
    fallback = (
        df[is_user & has_question & not_attachment]
        .sort_values(["id", "question_id", "ordre_message_question"])
        .groupby(["id", "question_id"])
        .agg(
            datetime_question_fallback=("heure_message_dt", "first"),
            question_fallback=("message", "first"),
        )
        .reset_index()
    )

    result = primary.merge(fallback, how="outer", on=["id", "question_id"])

    result["datetime_question"] = result["datetime_question"].fillna(
        result["datetime_question_fallback"]
    )
    result["question"] = result["question"].fillna(result["question_fallback"])

    return result[["id", "question_id", "datetime_question", "question"]]


# =============================================================================
# Feedback positif après intervention agent
# =============================================================================

def agent_positive_feedback_analysis(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte si l'utilisateur exprime un feedback positif après une intervention
    d'un agent humain, pour chaque (id, question_id).
    """
    df = questions.copy().sort_values(["id", "question_id", "ordre_message_question"])

    df["human_agent_has_spoken_before"] = (
        (df["speaker_type"] == "human_agent")
        .groupby([df["id"], df["question_id"]])
        .cummax()
        .shift(1)
        .fillna(False)
    )

    df["user_positive_feedback_after_agent"] = (
        (df["speaker_type"] == "user")
        & df["human_agent_has_spoken_before"]
        & df["message"].fillna("").str.contains(utils.PATTERN_FEEDBACK_POSITIF_USER, regex=True)
    )

    return (
        df.groupby(["id", "question_id"])
        .agg(agent_positive_feedback=("user_positive_feedback_after_agent", "max"))
        .reset_index()
    )


# =============================================================================
# Pipeline principal d'analyse
# =============================================================================

# Correspondance entre statut question et libellé lisible
QUESTION_STATUS_LABELS = {
    "ticket_created": "Ticket créé",
    "redirected_to_form": "Redirection vers formulaire",
    "resolved_by_agent": "Résolution par agent",
    "not_resolved": "Agent non en mesure d'aider",
    "unknown": "Inconnu",
}

# Colonnes booléennes à forcer à False si NaN
BOOL_COLS = [
    "bot_a_repondu",
    "feedback_positif_utilisateur_apres_reponse_bot",
    "escalade_proposee",
    "escalade_effective",
    "agent_a_pu_repondre",
    "feedback_positif_utilisateur_apres_reponse_agent",
    "csat_answered",
]

# Colonnes finales à exporter, dans l'ordre souhaité
FINAL_COLS = [
    "id_conv",
    "question_id",
    "nb_msg_question",
    "date_question",
    "heure_question",
    "question",
    "bot_a_repondu",
    "feedback_positif_utilisateur_apres_reponse_bot",
    "escalade_proposee",
    "escalade_effective",
    "agent_a_pu_repondre",
    "comment_agent_a_repondu",
    "feedback_positif_utilisateur_apres_reponse_agent",
    "csat_answered",
    "csat_feedback",
    "transcript_lisible",
]


def build_analysis(
    conversations: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Pipeline complet d'analyse des conversations Butler sur une période donnée.

    Étapes :
    1. Filtrage par date
    2. Parsing du transcript → messages individuels
    3. Segmentation en questions
    4. Métriques, contenu, réponses bot/agent, escalades, CSAT
    5. Merge et mise en forme finale

    Retourne un DataFrame avec une ligne par (conversation, question).
    """
    df = conversations.copy()

    # 1. Filtrage par date
    df["started_dt"] = parse_started_date(df)
    df["date"] = df["started_dt"].dt.date
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    # 2. Transcript & hc → messages individuels
    hotels_code = utils.get_hotel_code(df)
    transcript = utils.get_transcript(df)

    # 3. Segmentation en questions (enrichit le transcript avec speaker_type, question_id, etc.)
    questions = utils.get_questions(transcript)

    # 4. Calculs par question
    q_metrics = utils.question_gen_metrics(questions)
    q_content = get_question_content(questions)
    bot_answer = utils.answer_analysis(questions)
    escalation = utils.escalation_analysis(questions)
    agent_reply = utils.agent_reply_analysis(questions)
    agent_feedback = agent_positive_feedback_analysis(questions)
    csat = utils.survey_evaluation(df)
    readable_transcript = build_readable_transcript(questions)

    # 5. Merge de toutes les analyses
    result = q_metrics
    for other, key in [
        (q_content, ["id", "question_id"]),
        (bot_answer, ["id", "question_id"]),
        (escalation, ["id", "question_id"]),
        (agent_reply, ["id", "question_id"]),
        (agent_feedback, ["id", "question_id"]),
        (csat, ["id"]),
        (readable_transcript, ["id"]),
        (hotels_code, ["id"]),
    ]:
        result = result.merge(other, how="left", on=key)

    # 6. Mise en forme
    result["date_question"] = pd.to_datetime(result["datetime_question"]).dt.date
    result["heure_question"] = pd.to_datetime(result["datetime_question"]).dt.time

    result["agent_a_pu_repondre"] = result["question_status"].isin([
        "ticket_created", "redirected_to_form", "resolved_by_agent",
    ])
    result["comment_agent_a_repondu"] = result["question_status"].map(QUESTION_STATUS_LABELS)

    result["csat_answered"] = result["has_survey"].fillna(False)
    result["csat_feedback"] = result["survey_feedback"].replace({
        "thumbs_up": "positive",
        "thumbs_down": "negative",
        "none": "none",
        False: "none",
    })

    # 7. Renommage et sélection des colonnes finales
    result = result.rename(columns={
        "id": "id_conv",
        "bot_answer": "bot_a_repondu",
        "bot_answer_satisfying": "feedback_positif_utilisateur_apres_reponse_bot",
        "escalation_offered": "escalade_proposee",
        "escalation_effective": "escalade_effective",
        "agent_positive_feedback": "feedback_positif_utilisateur_apres_reponse_agent",
    })

    result = result[FINAL_COLS]

    for col in BOOL_COLS:
        result[col] = result[col].fillna(False)

    return result


# =============================================================================
# Interface Streamlit
# =============================================================================

REQUIRED_COLS = ["id", "started", "transcript", "hotelCode", "customerHandle", "assignee"]

export_conv = st.file_uploader("Export Conversations", type=["xlsx", "csv"])

if export_conv:
    conversations = read_file(export_conv)

    missing_cols = [col for col in REQUIRED_COLS if col not in conversations.columns]
    if missing_cols:
        st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
        st.stop()

    conv_dates = parse_started_date(conversations)
    min_date = conv_dates.min().date()
    max_date = conv_dates.max().date()

    start_date = st.date_input("Date de début d'analyse", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Date de fin d'analyse", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("La date de début ne peut pas être après la date de fin.")
        st.stop()

    st.success("Fichier chargé")
    st.write("Aperçu des conversations")
    st.dataframe(conversations.head())

    if st.button("Lancer l'analyse"):
        result = build_analysis(conversations, start_date, end_date)

        st.success("Analyse terminée")
        st.write("Résultat")
        st.dataframe(result)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="questions_analysis")
        output.seek(0)

        st.download_button(
            label="Télécharger les résultats",
            data=output,
            file_name="questions_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
