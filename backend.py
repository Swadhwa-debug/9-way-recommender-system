import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
from collections import defaultdict
#from surprise import NMF, Reader, Dataset, accuracy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EMBEDDING_SIZE = 16  # 
# Add these global variables at the top
REGRESSION_MODEL = None
CLASSIFICATION_MODEL = None
CLASSIFICATION_LABEL_ENCODER = None
RATING_DF_FIXED = None
USER_EMB_FIXED = None
ITEM_EMB_FIXED = None
# Add to the global variables section at the top
COURSE_BOW_DF = None
COURSE_VECTORS = None
# Define a random state
rs = 42

# Add these URLs to the existing URLs
user_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
item_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"
RATINGS_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"

# Load fixed data only once
def load_fixed_data():
    global RATING_DF_FIXED, USER_EMB_FIXED, ITEM_EMB_FIXED
    try:
        # Explicitly reset to ensure no caching issues
        RATING_DF_FIXED = None
        USER_EMB_FIXED = None
        ITEM_EMB_FIXED = None
        
        # Load datasets with explicit verification
        RATING_DF_FIXED = pd.read_csv(RATINGS_URL)
        USER_EMB_FIXED = pd.read_csv(user_emb_url)
        ITEM_EMB_FIXED = pd.read_csv(item_emb_url)
        
        # Validate datasets
        if USER_EMB_FIXED.empty:
            raise ValueError("USER_EMB_FIXED is empty")
        if ITEM_EMB_FIXED.empty:
            raise ValueError("ITEM_EMB_FIXED is empty")
            
        logger.info("Fixed data loaded successfully")
        logger.info(f"USER_EMB_FIXED shape: {USER_EMB_FIXED.shape}")
        logger.info(f"ITEM_EMB_FIXED shape: {ITEM_EMB_FIXED.shape}")
        logger.info(f"RATING_DF_FIXED shape: {RATING_DF_FIXED.shape}")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading fixed data: {str(e)}")
        # Create empty but valid structures
        USER_EMB_FIXED = pd.DataFrame(columns=['user'] + [f'UFeature{i}' for i in range(16)])
        ITEM_EMB_FIXED = pd.DataFrame(columns=['item'] + [f'CFeature{i}' for i in range(16)])
        RATING_DF_FIXED = pd.DataFrame(columns=['user', 'item', 'rating'])
        logger.info("Created fallback empty datasets")

# Update these functions in backend.py

def load_bow():
    """Load and cache the bag-of-words representation of courses"""
    global COURSE_BOW_DF
    if COURSE_BOW_DF is None:
        url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
        COURSE_BOW_DF = pd.read_csv(url)
        
        # Convert all bow columns to numeric
        bow_columns = COURSE_BOW_DF.columns[2:]  # Skip doc_index and doc_id
        for col in bow_columns:
            COURSE_BOW_DF[col] = pd.to_numeric(COURSE_BOW_DF[col], errors='coerce')
        
        # Fill NaN values with 0
        COURSE_BOW_DF[bow_columns] = COURSE_BOW_DF[bow_columns].fillna(0)
        
        logger.info("Loaded and converted BOW data")
    return COURSE_BOW_DF

def train_regression_with_embeddings():
    global REGRESSION_MODEL
    load_fixed_data()
    
    # Merge data
    merged_df = pd.merge(RATING_DF_FIXED, USER_EMB_FIXED, how='left', on='user').fillna(0)
    merged_df = pd.merge(merged_df, ITEM_EMB_FIXED, how='left', on='item').fillna(0)
    
    # Create feature vectors - CORRECTED to use concatenation
    u_features = [f"UFeature{i}" for i in range(EMBEDDING_SIZE)]
    c_features = [f"CFeature{i}" for i in range(EMBEDDING_SIZE)]
    
    # Get feature vectors
    user_embs = merged_df[u_features].values
    course_embs = merged_df[c_features].values
    
    # Concatenate embeddings (creates 32D features)
    X = np.concatenate([user_embs, course_embs], axis=1)
    y = merged_df['rating']
    
    # Train Ridge regression
    model = Ridge(alpha=1.0, random_state=rs)
    model.fit(X, y)
    REGRESSION_MODEL = model
    logger.info(f"Trained regression model with {X.shape[1]} features")

def regression_with_embeddings_recommendations(user_id, top_k):
    global REGRESSION_MODEL
    try:
        logger.info(f"Starting regression recs for user: {user_id}")
        load_fixed_data()
        
        if REGRESSION_MODEL is None:
            train_regression_with_embeddings()
        
        ratings_df = load_ratings()
        
        # Get user embedding
        if user_id in USER_EMB_FIXED['user'].values:
            user_emb = USER_EMB_FIXED.loc[USER_EMB_FIXED['user'] == user_id, 
                                       [f"UFeature{i}" for i in range(EMBEDDING_SIZE)]].values[0]
        else:
            user_courses = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
            user_emb = generate_embeddings_for_new_user(user_id, user_courses)
        
        # Get unrated courses
        rated_courses = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
        candidate_courses = ITEM_EMB_FIXED[~ITEM_EMB_FIXED['item'].isin(rated_courses)]
        
        # Predict ratings
        recommendations = {}
        for idx, row in candidate_courses.iterrows():
            course_id = row['item']
            course_emb = row[[f"CFeature{i}" for i in range(EMBEDDING_SIZE)]].values
            combined = np.concatenate([user_emb, course_emb])
            
            # Verify feature dimension
            if len(combined) != 32:
                logger.error(f"Feature dimension mismatch: expected 32, got {len(combined)}")
                continue
                
            pred = REGRESSION_MODEL.predict([combined])[0]
            recommendations[course_id] = pred
        
        return dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k])
    
    except Exception as e:
        logger.error(f"REGRESSION ERROR: {str(e)}", exc_info=True)
        return {}
    
def is_user_in_fixed_embeddings(user_id):
    """Safely check if user exists in fixed embeddings"""
    if USER_EMB_FIXED is None or USER_EMB_FIXED.empty:
        return False
    return user_id in USER_EMB_FIXED['user'].values




    




def train_classification_with_embeddings():
    global CLASSIFICATION_MODEL, CLASSIFICATION_LABEL_ENCODER
    logger.info("🚀 Training Classification with Embeddings model...")
    start_time = time.time()
    load_fixed_data()
    
    # Merge data
    merged_df = pd.merge(RATING_DF_FIXED, USER_EMB_FIXED, how='left', on='user').fillna(0)
    merged_df = pd.merge(merged_df, ITEM_EMB_FIXED, how='left', on='item').fillna(0)
    
    # Create feature vectors - CORRECTED to use concatenation
    u_features = [f"UFeature{i}" for i in range(EMBEDDING_SIZE)]
    c_features = [f"CFeature{i}" for i in range(EMBEDDING_SIZE)]
    
    # Get feature vectors
    user_embs = merged_df[u_features].values
    course_embs = merged_df[c_features].values
    
    # Concatenate embeddings (creates 32D features)
    X = np.concatenate([user_embs, course_embs], axis=1)
    y = merged_df['rating']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=rs)
    model.fit(X, y_encoded)
    CLASSIFICATION_MODEL = model
    CLASSIFICATION_LABEL_ENCODER = le
    logger.info(f"✅ Classification model trained in {time.time() - start_time:.2f}s | Features: {X.shape[1]}")

def classification_with_embeddings_recommendations(user_id, top_k):
    global CLASSIFICATION_MODEL, CLASSIFICATION_LABEL_ENCODER
    try:
        logger.info(f"Starting classification recs for user: {user_id}")
        load_fixed_data()
        
        if CLASSIFICATION_MODEL is None:
            train_classification_with_embeddings()
        
        ratings_df = load_ratings()
        
        # Get user embedding
        if is_user_in_fixed_embeddings(user_id):
            user_emb = USER_EMB_FIXED.loc[USER_EMB_FIXED['user'] == user_id, 
                                       [f"UFeature{i}" for i in range(EMBEDDING_SIZE)]].values[0]
        else:
            user_courses = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
            user_emb = generate_embeddings_for_new_user(user_id, user_courses)
        
        # Get unrated courses
        rated_courses = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
        candidate_courses = ITEM_EMB_FIXED[~ITEM_EMB_FIXED['item'].isin(rated_courses)]
        
        # Predict ratings
        recommendations = {}
        for idx, row in candidate_courses.iterrows():
            course_id = row['item']
            course_emb = row[[f"CFeature{i}" for i in range(EMBEDDING_SIZE)]].values
            combined = np.concatenate([user_emb, course_emb])
            
            # Verify feature dimension
            if len(combined) != 32:
                logger.error(f"Feature dimension mismatch: expected 32, got {len(combined)}")
                continue
                
            probas = CLASSIFICATION_MODEL.predict_proba([combined])[0]
            expected_rating = 0
            for i, class_val in enumerate(CLASSIFICATION_LABEL_ENCODER.classes_):
                expected_rating += class_val * probas[i]
            recommendations[course_id] = expected_rating
        
        return dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k])
    
    except Exception as e:
        logger.error(f"CLASSIFICATION ERROR: {str(e)}", exc_info=True)
        return {}
        

# URL for test‐user ratings used by the PCA‐based model
TEST_USER_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
)

# Globally cached DataFrames
TEST_USERS_DF = pd.read_csv(TEST_USER_URL)[["user", "item"]]
USER_PROFILE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
)
USER_PROFILE_DF = pd.read_csv(USER_PROFILE_URL)
FEATURE_NAMES = list(USER_PROFILE_DF.columns[1:])

# Standardize the profile features
scaler = StandardScaler()
USER_PROFILE_DF[FEATURE_NAMES] = scaler.fit_transform(USER_PROFILE_DF[FEATURE_NAMES])
CLUSTER_MODEL = None
def load_kmeans_model(n_clusters: int = 10):
    """Fit (and cache) a KMeans model on standardized USER_PROFILE_DF features."""
    global CLUSTER_MODEL
    if CLUSTER_MODEL is None:
        X = USER_PROFILE_DF[FEATURE_NAMES].values  # standardized already
        CLUSTER_MODEL = KMeans(n_clusters=n_clusters, random_state=42)
        CLUSTER_MODEL.fit(X)
    return CLUSTER_MODEL
def get_user_cluster_df():
    """Return a DataFrame with columns ['user', 'cluster'] for all users."""
    model = load_kmeans_model()
    labels = model.labels_
    return pd.DataFrame({"user": USER_PROFILE_DF["user"], "cluster": labels})

models = ("Course Similarity",
         
          "Clustering",
          "Clustering with PCA",
          "KNN",
          #"NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

# Data caching for better performance
SIM_MATRIX = None
COURSE_VECTORS = None
TITLE_MAP = None
IDX_ID_DICT = None
ID_IDX_DICT = None

# Global ratings dataframe
RATINGS_DF = pd.read_csv(RATINGS_URL)

# Neural Network model cache
NN_MODEL = None
USER_IDX_MAP = None
COURSE_IDX_MAP = None
NN_MIN_RATING = None
NN_MAX_RATING = None

def load_ratings():
    return RATINGS_DF.copy()

def load_course_sims():
    return pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/sim.csv")

def load_courses():
    df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df



def load_course_vectors():
    """Load and cache course vectors"""
    global COURSE_VECTORS
    if COURSE_VECTORS is None:
        bow_df = load_bow()  # This will ensure COURSE_BOW_DF is loaded
        
        # Create a pivot table only if we have data
        if not bow_df.empty:
            COURSE_VECTORS = bow_df.pivot_table(
                index='doc_index',
                columns='token',
                values='bow',
                fill_value=0
            )
        else:
            # Create empty DataFrame as fallback
            COURSE_VECTORS = pd.DataFrame()
            logger.error("BOW data is empty, created empty course vectors")
            
    return COURSE_VECTORS

def get_title_map():
    """Load and cache course title mapping"""
    global TITLE_MAP
    if TITLE_MAP is None:
        courses = load_courses()
        TITLE_MAP = courses.set_index('COURSE_ID')['TITLE'].to_dict()
    return TITLE_MAP

def get_doc_dicts():
    """Load and cache document mappings"""
    global IDX_ID_DICT, ID_IDX_DICT
    if IDX_ID_DICT is None or ID_IDX_DICT is None:
        bow_df = load_bow()
        
        # Check if we have valid data
        if bow_df.empty:
            logger.error("BOW data is empty, creating empty mappings")
            IDX_ID_DICT = {}
            ID_IDX_DICT = {}
        else:
            grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
            IDX_ID_DICT = grouped_df[['doc_id']].to_dict()['doc_id']
            ID_IDX_DICT = {v: k for k, v in IDX_ID_DICT.items()}
            
    return IDX_ID_DICT, ID_IDX_DICT

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix, sim_threshold=0.0, top_k=None):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        if enrolled_course not in id_idx_dict:
            continue
        enrolled_idx = id_idx_dict[enrolled_course]
        for candidate_course in unselected_course_ids:
            if candidate_course not in id_idx_dict:
                continue
            candidate_idx = id_idx_dict[candidate_course]
            sim = sim_matrix[enrolled_idx][candidate_idx]
            # Only add if it meets the threshold
            if sim >= sim_threshold:
                if candidate_course not in res or sim > res[candidate_course]:
                    res[candidate_course] = sim
    
    # Sort and limit results if top_k is provided
    sorted_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    if top_k is not None:
        sorted_res = dict(list(sorted_res.items())[:top_k])
    
    return sorted_res

def create_user_profile(user_id):
    """Create a user profile vector from enrolled courses.
    Always returns a vector of length = number of features (FEATURE_NAMES)."""
    ratings = load_ratings()
    course_vectors = load_course_vectors()
    _, id_idx_dict = get_doc_dicts()

    user_ratings = ratings[ratings['user'] == user_id]
    profile = np.zeros(len(FEATURE_NAMES))  # ensure correct dimensionality
    total_rating = 0

    for _, row in user_ratings.iterrows():
        course_id = row['item']
        rating = row['rating']
        if course_id in id_idx_dict:
            course_idx = id_idx_dict[course_id]
            if course_idx in course_vectors.index:
                course_vec = course_vectors.loc[course_idx].values
                # truncate or pad course_vec to match FEATURE_NAMES length
                if len(course_vec) > len(FEATURE_NAMES):
                    course_vec = course_vec[:len(FEATURE_NAMES)]
                elif len(course_vec) < len(FEATURE_NAMES):
                    course_vec = np.pad(course_vec, (0, len(FEATURE_NAMES) - len(course_vec)))
                profile += rating * course_vec
                total_rating += rating

    if total_rating > 0:
        profile /= total_rating

    return profile  # length = len(FEATURE_NAMES)


from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def user_profile_recommendations(user_id, sim_threshold=0.0, top_courses=None):
    # Load data
    course_vectors = load_courses()
    idx_id_dict, id_idx_dict = get_doc_dicts()
    ratings = load_ratings()
    user_ratings = ratings[ratings['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].tolist()

    # Build user profile vector (weighted by ratings)
    user_profile = create_user_profile(user_id).reshape(1, -1)

    # Collect candidate courses (not already enrolled)
    all_courses = set(idx_id_dict.values())
    candidate_courses = [cid for cid in all_courses if cid not in enrolled_course_ids]

    # Build candidate matrix
    candidate_indices = [id_idx_dict[cid] for cid in candidate_courses if cid in id_idx_dict]
    candidate_matrix = course_vectors.loc[candidate_indices].values

    # Compute raw dot product scores (no normalization)
    scores = candidate_matrix.dot(user_profile.flatten())

    # Map scores back to course IDs
    recommendations = {
        cid: float(score) for cid, score in zip(candidate_courses, scores) if score >= sim_threshold
    }

    # Sort by score (descending)
    sorted_recommendations = dict(
        sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    )

    # Limit to top_courses if specified
    if top_courses is not None:
        sorted_recommendations = dict(list(sorted_recommendations.items())[:top_courses])

    return sorted_recommendations


    


def clustering_recommendations(user_id, top_courses=10, pop_threshold=10):
    # Load data
    ratings = load_ratings()
    enrolled = ratings[ratings['user'] == user_id]['item'].tolist()
    course_titles = get_title_map()
    course_vectors = load_course_vectors()
    _, id_idx_dict = get_doc_dicts()

    # Load pre-fitted KMeans model on user profiles
    model = load_kmeans_model()
    user_profile_vec = USER_PROFILE_DF.loc[USER_PROFILE_DF["user"] == user_id, FEATURE_NAMES].values
    if user_profile_vec.size == 0:
        user_profile_vec = create_user_profile(user_id).reshape(1, -1)

    # Find cluster assignment
    cluster_id = model.predict(user_profile_vec)[0]

    # Build cluster popularity table
    cluster_df = get_user_cluster_df()
    ratings_labeled = ratings.merge(cluster_df, on="user", how="left")
    enrolls = (
        ratings_labeled
        .assign(count=1)
        .groupby(["cluster", "item"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "enrollments"})
    )

    # Get popular courses in this cluster
    cluster_courses = enrolls[(enrolls["cluster"] == cluster_id) & (enrolls["enrollments"] >= pop_threshold)]

    # Filter out courses the user already took
    unseen = cluster_courses[~cluster_courses["item"].isin(enrolled)]

    # Get cluster centroid
    centroid = model.cluster_centers_[cluster_id]

    # Build DataFrame with popularity + course-specific distance
    results = []
    for _, row in unseen.iterrows():
        course_id = row['item']
        if course_id in id_idx_dict:
            course_idx = id_idx_dict[course_id]
            if course_idx in course_vectors.index:
                course_vec = course_vectors.loc[course_idx].values
                course_distance = np.linalg.norm(course_vec - centroid)
            else:
                course_distance = None
        else:
            course_distance = None

        results.append({
            "USER": user_id,
            "COURSE_ID": course_id,
            "COURSE_TITLE": course_titles.get(course_id, "Unknown"),
            "Cluster ID": cluster_id,
            "Popularity": int(row['enrollments']),
            "Distance to Centroid": course_distance
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values(
        by=["Popularity", "Distance to Centroid"],
        ascending=[False, True]
    ).head(top_courses)





def pca_clustering_recommendations(n_components, n_clusters, pop_threshold):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("km", KMeans(n_clusters=n_clusters, random_state=123))
    ])
    X = USER_PROFILE_DF[FEATURE_NAMES].values
    pipe.fit(X)

    labels = pipe.named_steps["km"].labels_
    cluster_df = pd.DataFrame({"user": USER_PROFILE_DF["user"], "cluster": labels})

    ratings = load_ratings()
    ratings_labeled = ratings.merge(cluster_df, on="user", how="left")
    enrolls = (
        ratings_labeled
        .assign(count=1)
        .groupby(["cluster", "item"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "enrollments"})
    )

    course_titles = get_title_map()
    course_vectors = load_course_vectors()
    _, id_idx_dict = get_doc_dicts()

    results = []
    for _, row in enrolls.iterrows():
        cluster_id = row["cluster"]
        course_id = row["item"]
        if course_id in id_idx_dict:
            course_idx = id_idx_dict[course_id]
            if course_idx in course_vectors.index:
                course_vec = course_vectors.loc[course_idx].values.reshape(1, -1)
                course_pca_vec = pipe.named_steps["pca"].transform(
                    pipe.named_steps["scaler"].transform(course_vec)
                )
                centroid = pipe.named_steps["km"].cluster_centers_[cluster_id]
                course_distance = np.linalg.norm(course_pca_vec[0] - centroid)
            else:
                course_distance = None
        else:
            course_distance = None

        results.append({
            "Cluster ID": cluster_id,
            "COURSE_ID": course_id,
            "COURSE_TITLE": course_titles.get(course_id, "Unknown"),
            "Popularity": int(row['enrollments']),
            "Distance to Centroid": course_distance
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values(
        by=["Cluster ID", "Popularity", "Distance to Centroid"],
        ascending=[True, False, True]
    )



def item_knn_recommendations(n_neighbors: int, top_k: int) -> tuple[dict, dict]:
    start_time = time.time()
    print(f"🚀 Starting KNN recommendations at {time.strftime('%H:%M:%S')}")
    ratings = load_ratings()
    ui_df = ratings.pivot(index='user', columns='item', values='rating').fillna(0)
    item_user = ui_df.values.T
    n_items = item_user.shape[0]
    print("Computing item-item similarity matrix...")
    sim_start = time.time()
    item_sim_matrix = cosine_similarity(item_user)
    print(f"  ✅ Similarity matrix computed in {time.time() - sim_start:.2f}s")
    np.fill_diagonal(item_sim_matrix, 0)
    topk_sim_matrix = np.zeros_like(item_sim_matrix)
    for i in range(n_items):
        top_indices = np.argpartition(item_sim_matrix[i], -n_neighbors)[-n_neighbors:]
        topk_sim_matrix[i, top_indices] = item_sim_matrix[i, top_indices]
    print("Computing prediction scores...")
    pred_start = time.time()
    user_item = ui_df.values
    prediction_scores = user_item @ topk_sim_matrix
    seen_mask = (user_item > 0)
    prediction_scores[seen_mask] = -10**9
    print(f"  ✅ Predictions computed in {time.time() - pred_start:.2f}s")
    print("Finding top recommendations...")
    topk_start = time.time()
    recs = {}
    score_dict = {}
    user_ids = ui_df.index.tolist()
    item_ids = ui_df.columns.tolist()
    for i, uid in enumerate(user_ids):
        user_row = prediction_scores[i]
        top_indices = np.argpartition(-user_row, top_k)[:top_k]
        sorted_indices = top_indices[np.argsort(-user_row[top_indices])]
        recs[uid] = [item_ids[idx] for idx in sorted_indices]
        score_dict[uid] = [float(user_row[idx]) for idx in sorted_indices]
    print(f"  ✅ Top-k selected in {time.time() - topk_start:.2f}s")
    print(f"🏁 Total KNN time: {time.time() - start_time:.2f} seconds")
    return recs, score_dict

def add_new_ratings(new_courses):
    global RATINGS_DF, NN_MODEL, USER_IDX_MAP, COURSE_IDX_MAP
    
    if not new_courses:
        return None
        
    ratings_df = load_ratings()
    new_id = ratings_df['user'].max() + 1
    new_ratings = pd.DataFrame({
        'user': [new_id] * len(new_courses),
        'item': new_courses,
        'rating': [3.0] * len(new_courses)
    })
    
    RATINGS_DF = pd.concat([ratings_df, new_ratings], ignore_index=True)
    
    # Reset neural network cache
    NN_MODEL = None
    USER_IDX_MAP = None
    COURSE_IDX_MAP = None
    
    logger.info(f"Added new user {new_id} with {len(new_courses)} courses")
    return new_id

# NMF Model functions
#NMF_MODEL = None
#NMF_TRAINSET = None

#def train_nmf_model(n_factors=32, n_epochs=50):
    """Train and cache the NMF model"""
    global NMF_MODEL, NMF_TRAINSET
    print("🚀 Training NMF model...")
    start_time = time.time()
    
    # Load and prepare data
    ratings_df = load_ratings()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Train model
    model = NMF(n_factors=n_factors, n_epochs=n_epochs, random_state=42, verbose=False)
    model.fit(trainset)
    
    NMF_MODEL = model
    NMF_TRAINSET = trainset
    print(f"✅ NMF trained in {time.time() - start_time:.2f}s | Factors: {n_factors} | Epochs: {n_epochs}")
    return model, trainset

#def nmf_recommendations(user_id, top_k):
    """Generate recommendations using NMF"""
    if NMF_MODEL is None or NMF_TRAINSET is None:
        raise ValueError("NMF model not trained! Call train_nmf_model first")
    
    # Get user's rated courses
    ratings_df = load_ratings()
    rated_courses = set(ratings_df[ratings_df['user'] == user_id]['item'])
    
    # Predict ratings for all items
    predictions = []
    for item_id in ratings_df['item'].unique():
        if item_id not in rated_courses:
            pred = NMF_MODEL.predict(user_id, item_id)
            predictions.append((item_id, pred.est))
    
    # Sort and get top-k
    predictions.sort(key=lambda x: x[1], reverse=True)
    return {item: score for item, score in predictions[:top_k]}

# Neural Network Model functions
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # User embedding and bias
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)
        
        # Item embedding and bias
        self.item_embedding = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(input_dim=num_items, output_dim=1)
        
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        
        # Optimized dot product calculation
        dot_user_item = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
        x = dot_user_item + user_bias + item_bias
        return tf.nn.sigmoid(x)

def process_dataset(raw_data):
    encoded_data = raw_data.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return (
        encoded_data,
        user_id2idx_dict,
        user_idx2id_dict,
        course_id2idx_dict,
        course_idx2id_dict
    )

def generate_train_test_datasets(dataset, scale=True):
    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])
    
    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Training: 80%, Validation: 10%, Testing: 10%
    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])
    
    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test, min_rating, max_rating

# Update the train_neural_network function
def train_neural_network(embedding_size=16, epochs=10):
    global NN_MODEL, USER_IDX_MAP, COURSE_IDX_MAP, NN_MIN_RATING, NN_MAX_RATING
    logger.info("🚀 Training Neural Network model...")
    start_time = time.time()
    
    # Load and process data
    ratings_df = load_ratings()
    logger.info(f"Training with {len(ratings_df)} ratings, {ratings_df['user'].nunique()} users")
    
    # Process dataset and get mappings
    (
        encoded_data, 
        user_id2idx_dict, 
        user_idx2id_dict, 
        course_id2idx_dict, 
        course_idx2id_dict
    ) = process_dataset(ratings_df)
    
    # Set global mappings
    USER_IDX_MAP = user_id2idx_dict
    COURSE_IDX_MAP = course_id2idx_dict
    
    # Generate datasets
    x_train, x_val, _, y_train, y_val, _, min_rating, max_rating = generate_train_test_datasets(encoded_data)
    
    # Build model
    num_users = len(user_id2idx_dict)
    num_items = len(course_id2idx_dict)
    model = RecommenderNet(num_users, num_items, embedding_size)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Train model
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val)
    )
    
    # Cache model and metadata
    NN_MODEL = model
    NN_MIN_RATING = min_rating
    NN_MAX_RATING = max_rating
    
    logger.info(f"✅ Neural Network trained in {time.time() - start_time:.2f}s")
    return history

# Update the neural_network_recommendations function
def neural_network_recommendations(user_id, top_k=10):
    if NN_MODEL is None:
        raise ValueError("Neural network not trained! Call train_neural_network first")
    
    # Check if user exists in mapping
    if user_id not in USER_IDX_MAP:
        logger.error(f"User ID {user_id} not found. Add user via add_new_ratings() first.")
        return {}
    
    user_idx = USER_IDX_MAP[user_id]
    
    # Get all course indices
    all_course_ids = list(COURSE_IDX_MAP.keys())
    course_idxs = [COURSE_IDX_MAP[c] for c in all_course_ids]
    
    # Batch prediction
    batch_size = 1000
    predictions = []
    for i in range(0, len(course_idxs), batch_size):
        batch_course_idxs = course_idxs[i:i+batch_size]
        batch_inputs = np.array([[user_idx, c_idx] for c_idx in batch_course_idxs])
        batch_preds = NN_MODEL.predict(batch_inputs, verbose=0).flatten()
        predictions.extend(batch_preds.tolist())
    
    # Get user's rated courses
    ratings_df = load_ratings()
    rated_courses = set(ratings_df[ratings_df['user'] == user_id]['item'])
    
    # Scale predictions
    scaled_predictions = np.array(predictions) * (NN_MAX_RATING - NN_MIN_RATING) + NN_MIN_RATING
    
    # Create recommendations
    recommendations = {}
    for i, course_id in enumerate(all_course_ids):
        if course_id not in rated_courses:
            recommendations[course_id] = scaled_predictions[i]
    
    # Get top-k
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return dict(sorted_recs)
EMBEDDING_SIZE = 16  # Defined at the top of backend.py

def generate_embeddings_for_new_user(user_id, courses):
    try:
        logger.info(f"Generating embedding for new user {user_id} with {len(courses)} courses")
        load_fixed_data()
        
        # Get fixed course embeddings for the selected courses
        course_embs = []
        for course_id in courses:
            # Check if course exists in fixed embeddings
            if course_id in ITEM_EMB_FIXED['item'].values:
                emb = ITEM_EMB_FIXED.loc[ITEM_EMB_FIXED['item'] == course_id, 
                                       [f"CFeature{i}" for i in range(EMBEDDING_SIZE)]].values[0]
                course_embs.append(emb)
        
        # Create final embedding
        if course_embs:
            user_embedding = np.mean(course_embs, axis=0)
            logger.info(f"Generated embedding from {len(course_embs)} courses")
        else:
            # Fallback: Use average of all course embeddings
            all_embs = ITEM_EMB_FIXED[[f"CFeature{i}" for i in range(EMBEDDING_SIZE)]].values
            user_embedding = np.mean(all_embs, axis=0)
            logger.info("Used global course average as embedding")
        
        logger.info(f"Generated embedding shape: {user_embedding.shape}")
        return user_embedding
        
    except Exception as e:
        logger.error(f"EMBEDDING GENERATION ERROR: {str(e)}", exc_info=True)
        return np.zeros(EMBEDDING_SIZE)
    
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    title_map = get_title_map()
    sim_matrix = load_course_sims().to_numpy()
    idx_id_dict, id_idx_dict = get_doc_dicts()
    users = []
    courses = []
    titles = []
    scores = []
    print(f"🔔 Predict function called for {model_name} at {time.strftime('%H:%M:%S')}")
    for user_id in user_ids:
        if model_name == models[0]:
            top_courses_param = params.get('top_courses', 10)  # Get top_courses parameter
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
    
    # Use the enhanced function with both threshold and top_k
            res = course_similarity_recommendations(
                   idx_id_dict, 
                   id_idx_dict, 
                  enrolled_course_ids, 
                  sim_matrix, 
                 sim_threshold=sim_threshold,  # Pass the threshold
                top_k=top_courses_param       # Pass the top courses limit
               )
    
            for course_id, score in res.items():
                users.append(user_id)
                courses.append(course_id)
                titles.append(title_map.get(course_id, "Unknown Course"))
                scores.append(score)            
    
      
        elif model_name == models[1]:
            user_id = user_ids[0]
            top_courses = params.get("top_courses", 10)
            pop_threshold = params.get("pop_threshold", 10)

            df = clustering_recommendations(user_id, top_courses=top_courses, pop_threshold=pop_threshold)

            for _, row in df.iterrows():
                users.append(row["USER"])
                courses.append(row["COURSE_ID"])
                titles.append(row["COURSE_TITLE"])
        # Put a readable combined score; frontend can split/format if desired
                scores.append(f"Pop:{row['Popularity']} | Dist:{row['Distance to Centroid']:.3f}")


 
        elif model_name == models[2]:
            n_comp = params["n_components"]
            n_clust = params["n_clusters"]
            pop_threshold = params["pop_threshold"]
            rec_dict = pca_clustering_recommendations(n_components=n_comp, n_clusters=n_clust, pop_threshold=pop_threshold)
            for uid in user_ids:
                for cid in rec_dict.get(uid, []):
                    users.append(uid)
                    courses.append(cid)
                    titles.append(get_title_map().get(cid, "Unknown Course"))
                    scores.append(None)
        elif model_name == models[3]:
            n_neighbors = params["n_neighbors"]
            top_k = params["top_k"]
            print(f"🔔 KNN predict called at {time.time()}")
            rec_dict, score_dict = item_knn_recommendations(n_neighbors=n_neighbors, top_k=top_k)
            for uid in user_ids:
                if uid in rec_dict:
                    for i, cid in enumerate(rec_dict.get(uid, [])):
                        users.append(uid)
                        courses.append(cid)
                        titles.append(get_title_map().get(cid, "Unknown Course"))
                        scores.append(score_dict[uid][i])
        
        elif model_name == models[4]:
            # Neural Network
            embedding_size = params.get("embedding_size", 16)
            epochs = params.get("epochs", 10)
            top_k = params.get("top_k", 10)
            
            # Train model if not already trained or parameters changed
            if (NN_MODEL is None or 
                NN_MODEL.embedding_size != embedding_size or 
                (hasattr(NN_MODEL, 'epochs') and NN_MODEL.epochs != epochs)):
                train_neural_network(embedding_size, epochs)
            
            # Generate recommendations
            for user_id in user_ids:
                try:
                    res = neural_network_recommendations(user_id, top_k)
                    for course_id, score in res.items():
                        users.append(user_id)
                        courses.append(course_id)
                        titles.append(title_map.get(course_id, "Unknown Course"))
                        scores.append(score)
                except Exception as e:
                    logger.error(f"Error generating NN recommendations for user {user_id}: {str(e)}")
        elif model_name == models[5]:
            top_k = params.get("top_k", 10)
            for user_id in user_ids:
                try:
                   res = regression_with_embeddings_recommendations(user_id, top_k)
                   for course_id, score in res.items():
                        
                        users.append(user_id)
                        courses.append(course_id)
                        titles.append(title_map.get(course_id, "Unknown Course"))
                        scores.append(score)
                except Exception as e:
                      logger.error(f"Regression embedding error for user {user_id}: {str(e)}")
    
        elif model_name == models[6]:  # Classification with Embedding Features
            top_k = params.get("top_k", 10)
            for user_id in user_ids:
                try:
                    res = classification_with_embeddings_recommendations(user_id, top_k)
                    for course_id, score in res.items():
                        users.append(user_id)
                        courses.append(course_id)
                        titles.append(title_map.get(course_id, "Unknown Course"))
                        scores.append(score)
                except Exception as e:
                    logger.error(f"Classification embedding error for user {user_id}: {str(e)}")

    res_df = pd.DataFrame({
        'USER': users,
        'COURSE_ID': courses,
        'COURSE_TITLE': titles,
        'SCORE': scores
    })
    return res_df
