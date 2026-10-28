# streamlit run ui.py

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import tensorflow as tf

st.set_page_config(page_title="Fitness Hybrid Recommender", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

N_NEIGHBORS = 10  
GAN_N_SAMPLES = 10  

WORKOUT_TO_GOAL = {
    'Strength': ['Bodybuilding', 'Powerlifting', 'Powerbuilding', 'Muscle & Sculpting', 'Olympic Weightlifting'],
    'Cardio': ['Athletics', 'Bodyweight Fitness'],
    'HIIT': ['Athletics', 'Muscle & Sculpting', 'At-home & Calisthenics'],
    'Yoga': ['Bodyweight Fitness'],
}

PROGRAM_FEATURE_COLS = [
    'level_encoded', 'goal_encoded', 'program_length',
    'time_per_workout', 'number_of_exercises', 'sets',
]

FEATURE_COLS = [
    'Age', 'Gender_Encoded', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
    'Session_Duration (hours)', 'Calories_Burned', 'Workout_Type_Encoded',
    'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
    'Experience_Level', 'BMI',
]

DEFAULT_FEATURE_COLS = ['Avg_BPM', 'Resting_BPM', 'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)']

HYBRID_OPTIONS = {
    "CF + VAE": ('cf', 'vae', "Rating-pattern similarity (CF) blended with a learned latent-preference model (VAE)."),
    "CF + GAN": ('cf', 'gan', "Rating-pattern similarity (CF) blended with GAN-generated preference patterns."),
    "CBF + VAE": ('cbf', 'vae', "Program-attribute similarity (CBF) blended with a learned latent-preference model (VAE)."),
    "CBF + GAN": ('cbf', 'gan', "Program-attribute similarity (CBF) blended with GAN-generated preference patterns."),
    "CF + CBF": ('cf', 'cbf', "Rating-pattern similarity (CF) blended with program-attribute similarity (CBF)."),
}


class CollaborativeFiltering:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.item_similarity = None

    def fit(self, user_item_matrix):
        self.user_similarity = cosine_similarity(user_item_matrix)
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        self.user_item_matrix = user_item_matrix
        self.user_index_map = {uid: i for i, uid in enumerate(user_item_matrix.index)}
        self.program_index_map = {pid: i for i, pid in enumerate(user_item_matrix.columns)}

    def predict_user_based(self, user_id, program_id, k=5):
        if user_id not in self.user_index_map or program_id not in self.program_index_map:
            return 3.0
        user_index = self.user_index_map[user_id]
        program_index = self.program_index_map[program_id]
        user_similarities = self.user_similarity[user_index]
        similar_users_index = np.argsort(user_similarities)[::-1][1:k + 1]

        ratings, weights = [], []
        for idx in similar_users_index:
            rating = self.user_item_matrix.iloc[idx, program_index]
            if rating > 0 and user_similarities[idx] > 0:
                ratings.append(rating)
                weights.append(user_similarities[idx])

        if len(ratings) == 0 or sum(weights) == 0:
            return 3.0
        return np.average(ratings, weights=weights)

    def predict_item_based(self, user_id, program_id, k=5):
        if user_id not in self.user_index_map or program_id not in self.program_index_map:
            return 3.0
        user_index = self.user_index_map[user_id]
        program_index = self.program_index_map[program_id]
        item_similarities = self.item_similarity[program_index]
        similar_items_index = np.argsort(item_similarities)[::-1][1:k + 1]

        ratings, weights = [], []
        for idx in similar_items_index:
            rating = self.user_item_matrix.iloc[user_index, idx]
            if rating > 0 and item_similarities[idx] > 0:
                ratings.append(rating)
                weights.append(item_similarities[idx])

        if len(ratings) == 0 or sum(weights) == 0:
            return 3.0
        return np.average(ratings, weights=weights)

    def recommend(self, user_id, n_recommendations=5, method='hybrid'):
        if user_id not in self.user_index_map:
            return []
        user_index = self.user_index_map[user_id]
        user_ratings = self.user_item_matrix.iloc[user_index]
        unrated_programs = user_ratings[user_ratings == 0].index.tolist()

        predictions = []
        for program_id in unrated_programs:
            if method == 'user':
                pred = self.predict_user_based(user_id, program_id)
            elif method == 'item':
                pred = self.predict_item_based(user_id, program_id)
            else:
                pred = (self.predict_user_based(user_id, program_id) + self.predict_item_based(user_id, program_id)) / 2
            predictions.append((program_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class ContentBasedFiltering:
    def __init__(self):
        self.user_profiles = None
        self.program_features = None
        self.scaler = StandardScaler()

    def fit(self, users_df, programs_df, interactions_df):
        self.program_features = programs_df[['Program_ID'] + PROGRAM_FEATURE_COLS].copy()
        feature_matrix = self.program_features[PROGRAM_FEATURE_COLS].values
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        self.program_features[PROGRAM_FEATURE_COLS] = feature_matrix_scaled

        self.user_profiles = {}
        for user_id in users_df['User_ID']:
            user_interactions = interactions_df[interactions_df['User_ID'] == user_id]
            if len(user_interactions) > 0:
                weighted_features = np.zeros(len(PROGRAM_FEATURE_COLS))
                total_weight = 0
                for _, interaction in user_interactions.iterrows():
                    program_id = interaction['Program_ID']
                    rating = interaction['Rating']
                    program_features = self.program_features[
                        self.program_features['Program_ID'] == program_id
                    ][PROGRAM_FEATURE_COLS].values
                    if len(program_features) > 0:
                        weighted_features += program_features[0] * rating
                        total_weight += rating
                self.user_profiles[user_id] = weighted_features / total_weight if total_weight > 0 else np.zeros(len(PROGRAM_FEATURE_COLS))
            else:
                self.user_profiles[user_id] = np.zeros(len(PROGRAM_FEATURE_COLS))

    def predict(self, user_id, program_id):
        if user_id not in self.user_profiles:
            return 3.0
        user_profile = self.user_profiles[user_id]
        program_features = self.program_features[self.program_features['Program_ID'] == program_id]
        if len(program_features) == 0:
            return 3.0
        program_vector = program_features[PROGRAM_FEATURE_COLS].values[0]
        similarity = np.dot(user_profile, program_vector) / (
            np.linalg.norm(user_profile) * np.linalg.norm(program_vector) + 1e-8
        )
        return max(1.0, min(5.0, 3.0 + similarity * 2.0))

    def recommend(self, user_id, programs_df, interactions_df, n_recommendations=5):
        if user_id not in self.user_profiles:
            return []
        user_interactions = interactions_df[interactions_df['User_ID'] == user_id]['Program_ID'].tolist()
        unrated_programs = programs_df[~programs_df['Program_ID'].isin(user_interactions)]['Program_ID'].tolist()
        predictions = [(pid, self.predict(user_id, pid)) for pid in unrated_programs]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@st.cache_resource(show_spinner="Loading cleaned data and trained models...")
def load_artifacts():
    member_clean = pd.read_csv(os.path.join(DATA_DIR, "member_clean.csv"))
    program_clean = pd.read_csv(os.path.join(DATA_DIR, "program_clean.csv"))
    interactions_df = pd.read_csv(os.path.join(DATA_DIR, "interactions_df.csv"))

    user_item_matrix = pd.read_csv(os.path.join(DATA_DIR, "user_item_matrix.csv"), index_col=0)
    user_item_matrix.columns = user_item_matrix.columns.astype(int)

    with open(os.path.join(DATA_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "cf_model.pkl"), "rb") as f:
        cf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "cbf_model.pkl"), "rb") as f:
        cbf_model = pickle.load(f)

    vae_encoder = load_model(os.path.join(MODEL_DIR, "vae_encoder.keras"), custom_objects={"Sampling": Sampling}, compile=False)
    vae_decoder = load_model(os.path.join(MODEL_DIR, "vae_decoder.keras"), compile=False)
    with open(os.path.join(MODEL_DIR, "vae_meta.pkl"), "rb") as f:
        vae_meta = pickle.load(f)

    gan_generator = load_model(os.path.join(MODEL_DIR, "gan_generator.keras"), compile=False)
    with open(os.path.join(MODEL_DIR, "gan_meta.pkl"), "rb") as f:
        gan_meta = pickle.load(f)

    return {
        "member_clean": member_clean,
        "program_clean": program_clean,
        "interactions_df": interactions_df,
        "user_item_matrix": user_item_matrix,
        "encoders": encoders,
        "cf_model": cf_model,
        "cbf_model": cbf_model,
        "vae_encoder": vae_encoder,
        "vae_decoder": vae_decoder,
        "vae_meta": vae_meta,
        "gan_generator": gan_generator,
        "gan_meta": gan_meta,
    }


def estimate_defaults(member_clean, workout_type, experience_level):
    subset = member_clean[
        (member_clean['Workout_Type'] == workout_type) & (member_clean['Experience_Level'] == experience_level)
    ]
    if len(subset) == 0:
        subset = member_clean
    return subset[DEFAULT_FEATURE_COLS].median()


def build_new_user_features(member_clean, encoders, age, gender, weight, height,
                             workout_type, experience_level, frequency, session_duration):
    bmi = weight / (height ** 2)
    defaults = estimate_defaults(member_clean, workout_type, experience_level)

    row = {
        'Age': age,
        'Gender_Encoded': encoders['Gender'].transform([gender])[0],
        'Weight (kg)': weight,
        'Height (m)': height,
        'Avg_BPM': defaults['Avg_BPM'],
        'Resting_BPM': defaults['Resting_BPM'],
        'Session_Duration (hours)': session_duration,
        'Calories_Burned': defaults['Calories_Burned'],
        'Workout_Type_Encoded': encoders['Workout_Type'].transform([workout_type])[0],
        'Fat_Percentage': defaults['Fat_Percentage'],
        'Water_Intake (liters)': defaults['Water_Intake (liters)'],
        'Workout_Frequency (days/week)': frequency,
        'Experience_Level': experience_level,
        'BMI': bmi,
    }
    fitness_score = (
        frequency * 0.3 + session_duration * 0.2 + (defaults['Calories_Burned'] / 1000) * 0.3 + experience_level * 0.2
    )
    return row, bmi, fitness_score


def find_similar_users(member_clean, new_user_row, k=10):
    feature_matrix = member_clean[FEATURE_COLS].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    new_vec = scaler.transform(pd.DataFrame([new_user_row])[FEATURE_COLS].values)
    sims = cosine_similarity(new_vec, scaled)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return member_clean.iloc[top_idx]['User_ID'].values, sims[top_idx]


def cf_score_existing(cf_model, user_id, program_id):
    return (cf_model.predict_user_based(user_id, program_id) + cf_model.predict_item_based(user_id, program_id)) / 2


def cf_score_cold(user_item_matrix, neighbor_ids, neighbor_sims, program_id):
    if program_id not in user_item_matrix.columns:
        return 3.0
    ratings, weights = [], []
    for uid, sim in zip(neighbor_ids, neighbor_sims):
        if uid in user_item_matrix.index:
            r = user_item_matrix.loc[uid, program_id]
            if r > 0 and sim > 0:
                ratings.append(r)
                weights.append(sim)
    if not ratings:
        return 3.0
    return float(np.average(ratings, weights=weights))


def cbf_score_existing(cbf_model, user_id, program_id):
    return cbf_model.predict(user_id, program_id)


def build_cbf_pseudo_profile(cbf_model, neighbor_ids, neighbor_sims):
    weights = np.clip(neighbor_sims, 0, None)
    profiles, valid_weights = [], []
    for uid, w in zip(neighbor_ids, weights):
        prof = cbf_model.user_profiles.get(uid)
        if prof is not None:
            profiles.append(prof)
            valid_weights.append(w)
    if not profiles:
        return None
    valid_weights = np.array(valid_weights)
    if valid_weights.sum() == 0:
        valid_weights = np.ones_like(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    return np.average(np.vstack(profiles), axis=0, weights=valid_weights)


def cbf_score_cold(cbf_model, pseudo_profile, program_id):
    if pseudo_profile is None:
        return 3.0
    prog_row = cbf_model.program_features[cbf_model.program_features['Program_ID'] == program_id]
    if len(prog_row) == 0:
        return 3.0
    program_vector = prog_row[PROGRAM_FEATURE_COLS].values[0]
    similarity = np.dot(pseudo_profile, program_vector) / (
        np.linalg.norm(pseudo_profile) * np.linalg.norm(program_vector) + 1e-8
    )
    return max(1.0, min(5.0, 3.0 + similarity * 2.0))


def vae_reconstruct_vector(vae_encoder, vae_decoder, normalized_vector):
    """One forward pass through the VAE for a single (1, n_programs) normalized vector."""
    z_mean, z_log_var, z = vae_encoder(normalized_vector, training=False)
    reconstruction = vae_decoder(z, training=False)
    return reconstruction.numpy()[0]


def vae_score_existing_from_ctx(reconstruction, user_item_matrix, program_id):
    if reconstruction is None or program_id not in user_item_matrix.columns:
        return 3.0
    program_idx = list(user_item_matrix.columns).index(program_id)
    return reconstruction[program_idx] * 5.0


def build_vae_pseudo_reconstruction(vae_encoder, vae_decoder, user_item_matrix, neighbor_ids, neighbor_sims):
    valid_mask = np.isin(neighbor_ids, user_item_matrix.index)
    valid_ids = neighbor_ids[valid_mask]
    valid_sims = neighbor_sims[valid_mask]
    if len(valid_ids) == 0:
        return None
    weights = np.clip(valid_sims, 0, None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    rows = user_item_matrix.loc[valid_ids].values
    pseudo_vector = np.average(rows, axis=0, weights=weights)
    normalized = (pseudo_vector / 5.0).reshape(1, -1)
    return vae_reconstruct_vector(vae_encoder, vae_decoder, normalized)


def vae_score_cold(reconstruction, user_item_matrix, program_id):
    if reconstruction is None or program_id not in user_item_matrix.columns:
        return 3.0
    program_idx = list(user_item_matrix.columns).index(program_id)
    return reconstruction[program_idx] * 5.0


def build_gan_avg_preferences(gan_generator, latent_dim, n_samples=GAN_N_SAMPLES):
    """One batched forward pass through the GAN generator (user-agnostic, same for every mode)."""
    noise = np.random.normal(0, 1, (n_samples, latent_dim)).astype(np.float32)
    preds = gan_generator(noise, training=False).numpy()
    avg_preferences = preds.mean(axis=0)
    pref_min, pref_max = avg_preferences.min(), avg_preferences.max()
    return (avg_preferences - pref_min) / (pref_max - pref_min + 1e-8)


def gan_score_from_scaled(scaled_preferences, user_item_matrix, program_id):
    if scaled_preferences is None or program_id not in user_item_matrix.columns:
        return 3.0
    program_idx = list(user_item_matrix.columns).index(program_id)
    return 1 + scaled_preferences[program_idx] * 4


def compute_component_score(component, mode, program_id, art, ctx):
    """ctx carries whatever the current mode needs (user_id, pseudo profiles, precomputed vectors, etc.)."""
    if component == 'cf':
        if mode == 'existing':
            return cf_score_existing(art['cf_model'], ctx['user_id'], program_id)
        return cf_score_cold(art['user_item_matrix'], ctx['neighbor_ids'], ctx['neighbor_sims'], program_id)

    if component == 'cbf':
        if mode == 'existing':
            return cbf_score_existing(art['cbf_model'], ctx['user_id'], program_id)
        return cbf_score_cold(art['cbf_model'], ctx['cbf_pseudo_profile'], program_id)

    if component == 'vae':
        if mode == 'existing':
            return vae_score_existing_from_ctx(ctx['vae_reconstruction'], art['user_item_matrix'], program_id)
        return vae_score_cold(ctx['vae_reconstruction'], art['user_item_matrix'], program_id)

    if component == 'gan':
        return gan_score_from_scaled(ctx['gan_scaled_preferences'], art['user_item_matrix'], program_id)

    return 3.0


def recommend_existing_user(hybrid_key, art, user_id, n_recommendations=5):
    comp1, comp2, _ = HYBRID_OPTIONS[hybrid_key]
    user_item_matrix = art['user_item_matrix']
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    candidate_ids = user_ratings[user_ratings == 0].index.tolist()

    ctx = {'user_id': user_id}
    if 'vae' in (comp1, comp2):
        user_vector = user_item_matrix.loc[[user_id]].values / 5.0
        ctx['vae_reconstruction'] = vae_reconstruct_vector(art['vae_encoder'], art['vae_decoder'], user_vector)
    if 'gan' in (comp1, comp2):
        ctx['gan_scaled_preferences'] = build_gan_avg_preferences(art['gan_generator'], art['gan_meta']['latent_dim'])

    results = []
    for pid in candidate_ids:
        s1 = compute_component_score(comp1, 'existing', pid, art, ctx)
        s2 = compute_component_score(comp2, 'existing', pid, art, ctx)
        hybrid = 0.5 * s1 + 0.5 * s2
        results.append((pid, hybrid, comp1, s1, comp2, s2))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_recommendations]


def recommend_new_user(hybrid_key, art, age, gender, weight, height, workout_type,
                        experience_level, frequency, session_duration, n_recommendations=5):
    comp1, comp2, _ = HYBRID_OPTIONS[hybrid_key]
    member_clean = art['member_clean']
    program_clean = art['program_clean']
    user_item_matrix = art['user_item_matrix']

    new_user_row, bmi, fitness_score = build_new_user_features(
        member_clean, art['encoders'], age, gender, weight, height,
        workout_type, experience_level, frequency, session_duration,
    )
    neighbor_ids, neighbor_sims = find_similar_users(member_clean, new_user_row, k=N_NEIGHBORS)

    candidate_goals = WORKOUT_TO_GOAL.get(workout_type, [])
    candidate_ids = program_clean[program_clean['primary_goal'].isin(candidate_goals)]['Program_ID'].tolist()

    ctx = {'neighbor_ids': neighbor_ids, 'neighbor_sims': neighbor_sims}
    if 'cbf' in (comp1, comp2):
        ctx['cbf_pseudo_profile'] = build_cbf_pseudo_profile(art['cbf_model'], neighbor_ids, neighbor_sims)
    if 'vae' in (comp1, comp2):
        ctx['vae_reconstruction'] = build_vae_pseudo_reconstruction(
            art['vae_encoder'], art['vae_decoder'], user_item_matrix, neighbor_ids, neighbor_sims
        )
    if 'gan' in (comp1, comp2):
        ctx['gan_scaled_preferences'] = build_gan_avg_preferences(art['gan_generator'], art['gan_meta']['latent_dim'])

    results = []
    for pid in candidate_ids:
        s1 = compute_component_score(comp1, 'cold', pid, art, ctx)
        s2 = compute_component_score(comp2, 'cold', pid, art, ctx)
        hybrid = 0.5 * s1 + 0.5 * s2
        results.append((pid, hybrid, comp1, s1, comp2, s2))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_recommendations], bmi, fitness_score


LABELS = {'cf': 'CF', 'cbf': 'CBF', 'vae': 'VAE', 'gan': 'GAN'}


def render_recommendations(recs, program_clean):
    if not recs:
        st.warning("No matching programs found.")
        return

    st.subheader("Recommended Programs")
    for rank, (program_id, hybrid_score, comp1, s1, comp2, s2) in enumerate(recs, start=1):
        program_info = program_clean[program_clean['Program_ID'] == program_id].iloc[0]
        with st.container(border=True):
            top1, top2 = st.columns([4, 1])
            top1.markdown(f"**{rank}. {program_info['title']}**")
            top2.metric("Hybrid Score", f"{hybrid_score:.2f}")

            meta1, meta2, meta3 = st.columns(3)
            meta1.write(f"**Level:** {program_info['primary_level']}")
            meta2.write(f"**Goal:** {program_info['primary_goal']}")
            meta3.write(f"**Duration:** {program_info['program_length']} weeks")

            with st.expander("View more details"):
                st.write(program_info['description'])
                d1, d2, d3, d4 = st.columns(4)
                d1.write(f"**Equipment:** {program_info['equipment']}")
                d2.write(f"**Time/workout:** {program_info['time_per_workout']} min")
                d3.write(f"**Exercises:** {program_info['number_of_exercises']}")
                d4.write(f"**Sets:** {program_info['sets']}")

                s_col1, s_col2 = st.columns(2)
                s_col1.metric(f"{LABELS[comp1]} Score", f"{s1:.3f}")
                s_col2.metric(f"{LABELS[comp2]} Score", f"{s2:.3f}")


st.sidebar.header("Model Settings")
hybrid_key = st.sidebar.selectbox("Hybrid Model", list(HYBRID_OPTIONS.keys()))
st.sidebar.caption(HYBRID_OPTIONS[hybrid_key][2])
n_recs = st.sidebar.slider("Number of recommendations", 1, 10, 5)

st.title("Personalized Fitness Hybrid-Based Recommender System")
st.caption(f"Currently using **{hybrid_key}**")

try:
    art = load_artifacts()
except FileNotFoundError:
    st.error(
        "Cleaned data / trained models not found. Run the notebook's final "
        "\"Export Cleaned Data & Trained Models\" cell first — it saves everything "
        "needed under `data/` and `models/` next to this app."
    )
    st.stop()

member_clean = art['member_clean']
program_clean = art['program_clean']
user_item_matrix = art['user_item_matrix']

col_a, col_b, col_c = st.columns(3)
col_a.metric("Reference Users", user_item_matrix.shape[0])
col_b.metric("Programs", user_item_matrix.shape[1])
col_c.metric("Recorded Interactions", len(art['interactions_df']))

mode = st.radio(
    "What would you like to do?",
    ["Explore existing users in the dataset", "Get recommendations for a new user"],
    horizontal=True,
)

if mode == "Explore existing users in the dataset":
    with st.expander("Dataset overview"):
        ov1, ov2 = st.columns(2)
        ov1.write("**Workout type distribution**")
        ov1.bar_chart(member_clean['Workout_Type'].value_counts())
        ov2.write("**Program goal distribution**")
        ov2.bar_chart(program_clean['primary_goal'].value_counts())

    st.subheader("Select an existing user")
    rated_users = member_clean[member_clean['User_ID'].isin(user_item_matrix.index)]
    level_labels = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    user_options = rated_users[['User_ID', 'Gender', 'Age', 'Workout_Type', 'Experience_Level', 'BMI']].copy()
    user_options['label'] = user_options.apply(
        lambda r: (
            f"User {r.User_ID} : {r.Gender}, {r.Age} years old, {r.Workout_Type}, "
            f"{level_labels.get(r.Experience_Level, r.Experience_Level)}"
        ),
        axis=1,
    )
    label_to_id = dict(zip(user_options['label'], user_options['User_ID']))
    selected_label = st.selectbox("Existing user", user_options['label'].tolist())
    selected_user_id = label_to_id[selected_label]

    profile = user_options[user_options['User_ID'] == selected_user_id].iloc[0]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Age", int(profile['Age']))
    p2.metric("Workout Type", profile['Workout_Type'])
    p3.metric("Level", level_labels.get(profile['Experience_Level'], profile['Experience_Level']))
    p4.metric("BMI", f"{profile['BMI']:.1f}")

    if st.button("Get Recommendations", type="primary"):
        recs = recommend_existing_user(hybrid_key, art, selected_user_id, n_recommendations=n_recs)
        render_recommendations(recs, program_clean)

else:
    st.subheader("Your details")
    with st.form("user_profile_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=10, max_value=100, value=25)
        gender = c2.selectbox("Gender", art['encoders']['Gender'].classes_.tolist())
        experience_level = c3.selectbox(
            "Experience Level", [1, 2, 3],
            format_func=lambda x: {1: "Beginner", 2: "Intermediate", 3: "Advanced"}[x],
        )

        c4, c5, c6 = st.columns(3)
        weight = c4.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        height = c5.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        workout_type = c6.selectbox("Preferred Workout Type", art['encoders']['Workout_Type'].classes_.tolist())

        c7, c8 = st.columns(2)
        frequency = c7.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)
        session_duration = c8.number_input("Session Duration (hours)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)

        submitted = st.form_submit_button("Get Recommendations", type="primary")

    if submitted:
        recs, bmi, fitness_score = recommend_new_user(
            hybrid_key, art, age, gender, weight, height, workout_type,
            experience_level, frequency, session_duration, n_recommendations=n_recs,
        )
        m1, m2 = st.columns(2)
        m1.metric("BMI", f"{bmi:.1f}")
        m2.metric("Estimated Fitness Score", f"{fitness_score:.2f}")
        render_recommendations(recs, program_clean)