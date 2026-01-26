import streamlit as st
import pandas as pd
import time
import backend as backend
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
from backend import FEATURE_NAMES, USER_PROFILE_DF, TEST_USERS_DF
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Custom styling
# Custom styling
st.markdown("""
<style>
h1 {
    font-size: 2.8rem !important;
    width: 100% !important;
    max-width: 1200px !important;
    margin: 10px auto 25px auto !important;
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border-radius: 15px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
thead tr th {
    background-color: #2c3e50 !important;
    color: white !important;
    font-weight: bold;
    font-size: 1.2rem;
}
tbody tr:hover {
    background-color: #e3f2fd !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    height: 15px;
    border-radius: 10px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%);
    color: white;
}
.stButton>button {
    background: linear-gradient(45deg, #2193b0, #6dd5ed) !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 24px;
}

/* Add these for sidebar widget labels */
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label {
    color: white !important;
    font-weight: 500;
}

/* Optional: Make the slider values and tooltips more visible */
div[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
    color: white !important;
}

/* Style the slider track and thumb */
div[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #182848 !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
# Additional sidebar styling for widget labels
st.markdown("""
<style>
/* Target all widget labels in sidebar */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
# Add visible title to the main content area
st.title("🎓 Course Recommender System")
st.markdown("Discover personalized course recommendations based on your learning history")
st.markdown("""---""")

# Demo steps
st.markdown("""
### 🚀 Quick Start Guide:
1. **Select Courses**: Choose courses you've completed from the table below
2. **Create User**: Click "Add Ratings for New User" to create your profile  
3. **Choose Algorithm**: Select from 8 ML models in the sidebar
4. **Get Recommendations**: Click the recommendation button for your model
5. **Compare Results**: Try different algorithms to see varied suggestions
""")


# ------- Functions ------
@st.cache_data
def load_course_sims():
    return backend.load_course_sims()

@st.cache_data
def load_courses():
    return backend.load_courses()

@st.cache_data
def load_bow():
    return backend.load_bow()

# Initialize the app
def init__recommender_app():
    with st.spinner('Loading datasets...'):
        ratings_df = backend.load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        
        # Debug: show dataset sizes
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Courses", course_df.shape[0])
        with col2:
            st.metric("Ratings", ratings_df.shape[0])
            st.metric("Unique Users", ratings_df['user'].nunique())
        with col3:
            st.metric("Similarities", sim_df.shape[0])

    st.success('Datasets loaded successfully!')
    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        height=400
    )

    # Initialize session state variables
    if 'test_user_id' not in st.session_state:
        st.session_state.test_user_id = None
    if 'user_courses' not in st.session_state:
        st.session_state.user_courses = []
    if 'knn_state' not in st.session_state:
        st.session_state.knn_state = {
            'run_requested': False,
            'completed': False,
            'df': None
        }
    
    # Process selected courses
    if response["selected_rows"] is not None and len(response["selected_rows"]) > 0:
        results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
        results = results[['COURSE_ID', 'TITLE']]
        st.session_state.user_courses = results['COURSE_ID'].tolist()
    else:
        results = pd.DataFrame(columns=['COURSE_ID', 'TITLE'])
        st.session_state.user_courses = []
        
    st.subheader("Your selected courses: ")
    st.table(results)
    
    # Create Test User Section
    st.divider()
    st.subheader("Create Test User")
    
    # Add Ratings Button
    if st.button("➕ Add Ratings for New User", type="primary", key="add_ratings_btn"):
        if not st.session_state.user_courses:
            st.warning("Please select at least one course first!")
        else:
            with st.spinner("Creating new user profile..."):
                new_id = backend.add_new_ratings(st.session_state.user_courses)
            
            if new_id:
                st.session_state.test_user_id = new_id
                st.success(f"""
                ✅ Created new test user {new_id}!
                - User ID: `{new_id}`
                - Courses added: {len(st.session_state.user_courses)}
                - Default rating: 3 stars
                """)
            else:
                st.error("Failed to create new user")
    
    # Display current test user info
    if st.session_state.test_user_id:
        st.info(f"Current test user: **{st.session_state.test_user_id}**")
        if st.button("❌ Clear Test User", key="clear_user_btn"):
            st.session_state.test_user_id = None
            st.session_state.user_courses = []
            st.success("Test user cleared! Select new courses to create another user.")
    
    return results

# ------ UI ------
st.markdown("""
<style>
    .sidebar .stSelectbox label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Control Panel")
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox("Select model:", backend.models)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')

# Course similarity model
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider('Top courses', min_value=0, max_value=100, value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %', min_value=0, max_value=100, value=50, step=10)
    params = {'top_courses': top_courses, 'sim_threshold': course_sim_threshold}



# Clustering model
elif model_selection == backend.models[1]:
       cluster_no = st.sidebar.slider('Number of Clusters', min_value=2, max_value=50, value=20, step=1)
       top_courses = st.sidebar.slider('Courses per Cluster', min_value=1, max_value=10, value=3, step=1)
       params = {'cluster_num': cluster_no, 'top_courses': top_courses}
   
        
# Clustering with PCA
# Clustering with PCA
elif model_selection == backend.models[2]:
    st.sidebar.subheader("PCA + KMeans Parameters")
    n_components = st.sidebar.slider(
        "PCA Components",
        min_value=1,
        max_value=len(FEATURE_NAMES),
        value=5,
        step=1
    )
    n_clusters = st.sidebar.slider(
        "KMeans Clusters",
        min_value=2,
        max_value=20,
        value=9,
        step=1
    )
    pop_threshold = st.sidebar.slider(
        "Min Enrollments for Popularity",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )
    

    params = {
        "n_components": n_components,
        "n_clusters": n_clusters,
        "pop_threshold": pop_threshold,
        
    }


# KNN model
elif model_selection == backend.models[3]:
    n_neighbors = st.sidebar.slider("Neighbors (k)", 1, 50, 15, key="n_neighbors")
    top_k = st.sidebar.slider("Top-K recs/user", 1, 20, 5, key="top_k")
    params = {"n_neighbors": n_neighbors, "top_k": top_k}

# NMF model
#elif model_selection == backend.models[4]:
 #   st.sidebar.subheader("NMF Parameters")
  #  n_factors = st.sidebar.slider("Latent Factors", 10, 100, 32, step=1)
   # n_epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=5)
    #top_k = st.sidebar.slider("Top-K Recommendations", 1, 20, 5, step=1)
    #params = {"n_factors": n_factors, "n_epochs": n_epochs, "top_k": top_k}

# Neural Network model
elif model_selection == backend.models[4]:
    st.sidebar.subheader("Neural Network Parameters")
    embedding_size = st.sidebar.slider("Embedding Size", 8, 64, 16, step=8)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 10, step=5)
    top_k = st.sidebar.slider("Top-K Recommendations", 1, 20, 5, step=1)
    params = {"embedding_size": embedding_size, "epochs": epochs, "top_k": top_k}

elif model_selection == backend.models[5]:  # Regression with Embeddings
    top_k = st.sidebar.slider("Top-K Recommendations", 1, 20, 5, step=1)
    params = {"top_k": top_k}


elif model_selection == backend.models[6]:  # Classification with Embeddings
    top_k = st.sidebar.slider("Top-K Recommendations", 1, 20, 5, step=1)
    params = {"top_k": top_k}


# Prediction
st.sidebar.subheader('3. Prediction')

# Model-specific testing and prediction
if model_selection == backend.models[0]:
    if st.sidebar.button("🔎 Test Course Similarity"):
        ids = selected_courses_df["COURSE_ID"].tolist()
        if not ids:
            st.warning("Select at least one course above first.")
        else:
            user_id = backend.add_new_ratings(ids)
            threshold = params.get("sim_threshold", 50) / 100.0
            df = backend.predict("Course Similarity", [user_id], {"sim_threshold": threshold})
            if df.empty:
                st.error("No recommendations—try lowering the threshold.")
            else:
                st.subheader("📚 Recommended Courses")
                st.dataframe(df)


elif model_selection == backend.models[1]:
    if st.sidebar.button("🔎 Test Clustering"):
        ids = selected_courses_df["COURSE_ID"].tolist()
        if not ids:
            st.warning("Select at least one course first.")
        else:
            user_id = backend.add_new_ratings(ids)
            
            
            df = backend.predict("Clustering", [user_id], params)
            if df.empty:
                st.error("No recommendations found — try lowering the popularity threshold.")
            else:
                df = df.rename(columns={
                    "Cluster ID": "Cluster",
                })
                st.subheader("📊 Clustering Recommendations")
                st.dataframe(df.style.format({
                    "Popularity": "{:.0f}",
                    "Distance to Centroid": "{:.3f}",
                }))
                st.caption("ℹ️ Ranked by popularity in your cluster, then closeness to the cluster centroid.")


elif model_selection == backend.models[2]:
  
   

    if st.sidebar.button("▶️ Train & Recommend (PCA)", key="train_pca"):
        with st.spinner("Training PCA + KMeans model..."):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=params["n_components"])),
                ("km", KMeans(n_clusters=params["n_clusters"], random_state=123))
            ])
            X = USER_PROFILE_DF[FEATURE_NAMES].values
            labels = pipe.fit_predict(X)
            cluster_df = pd.DataFrame({"user": USER_PROFILE_DF["user"], "cluster": labels})
            test_labeled = TEST_USERS_DF.merge(cluster_df, on="user")
            enrolls = test_labeled.assign(count=1).groupby(["cluster","item"])["count"].sum().reset_index()
            popular = enrolls[enrolls["count"] >= params["pop_threshold"]].groupby("cluster")["item"].apply(set).to_dict()
            
            summary = pd.DataFrame([
                {"user": uid, "cluster": grp["cluster"].iloc[0], "n_recs": len(popular.get(grp["cluster"].iloc[0], []))}
                for uid, grp in test_labeled.groupby("user")
            ])
            
        st.success("PCA + KMeans training completed!")

        # Cluster statistics
        st.subheader("📊 Cluster Statistics")
        cluster_stats = cluster_df['cluster'].value_counts().sort_index()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", len(cluster_stats))
        with col2:
            st.metric("Total Users", cluster_stats.sum())
        with col3:
            st.metric("Avg Users/Cluster", f"{cluster_stats.mean():.1f}")
        
        # Show cluster sizes
        st.write("**Users per cluster:**")
        cluster_data = []
        for cluster_id, count in cluster_stats.items():
            cluster_data.append({"Cluster": cluster_id, "Users": count})
        cluster_size_df = pd.DataFrame(cluster_data)
        st.dataframe(cluster_size_df, hide_index=True)
        
        # Show distinctive courses in each cluster
        st.subheader("🎯 Distinctive Courses by Cluster")
        
        # Calculate distinctive courses
        cluster_course_counts = enrolls.groupby(['cluster', 'item'])['count'].sum().reset_index()
        global_popularity = enrolls.groupby('item')['count'].sum()
        
        distinctive_results = []
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            # Get courses for this cluster
            cluster_courses = cluster_course_counts[cluster_course_counts['cluster'] == cluster_id]
            
            # Calculate how distinctive each course is to this cluster
            distinctive_courses = []
            for _, row in cluster_courses.iterrows():
                course_id = row['item']
                cluster_count = row['count']
                global_count = global_popularity.get(course_id, 0)
                
                # Calculate relative popularity in this cluster vs globally
                if global_count > 0:
                    relative_popularity = cluster_count / global_count
                    distinctive_courses.append((course_id, cluster_count, relative_popularity))
            
            # Sort by distinctiveness and take top 5
            distinctive_courses.sort(key=lambda x: x[2], reverse=True)
            top_distinctive = distinctive_courses[:5]
            
            if top_distinctive:
                title_map = backend.get_title_map()
                for course_id, count, rel_pop in top_distinctive:
                    title = title_map.get(course_id, "Unknown Course")
                    distinctive_results.append({
                        "Cluster": cluster_id,
                        "Course": title,
                        "Enrollments": count,
                        "Relative Popularity": f"{rel_pop:.2f}"
                    })
        
        if distinctive_results:
            distinctive_df = pd.DataFrame(distinctive_results)
            
            # Display in a more readable format
            for cluster_id in sorted(distinctive_df['Cluster'].unique()):
                cluster_courses = distinctive_df[distinctive_df['Cluster'] == cluster_id]
                st.write(f"**Cluster {cluster_id}** ({cluster_stats.get(cluster_id, 0)} users):")
                
                course_list = []
                for _, row in cluster_courses.iterrows():
                    course_list.append(f"{row['Course']} (rel: {row['Relative Popularity']})")
                
                st.write(" • " + " • ".join(course_list))
                st.write("")  # Add some spacing
        else:
            st.info("No distinctive courses found. Try adjusting the popularity threshold.")
        
        # Model summary
        st.subheader("📈 Model Summary")
        st.write("""
        **PCA + KMeans Analysis** groups users into clusters based on their course enrollment patterns. 
        The 'Distinctive Courses' show which courses are over-represented in each cluster compared to the overall population.
        
        - **Relative Popularity**: How much more popular a course is in this cluster vs the entire dataset
        - **Higher values** indicate courses that are more characteristic of the cluster
        """)
        
        # Simple message for test user
        if st.session_state.get('test_user_id'):
            st.info("""
            **💡 For Personalized Recommendations:** 
            While PCA+KMeans shows user segments, for individual course recommendations try:
            - **User Profile Model**: Finds users similar to you
            - **KNN Model**: Item-based collaborative filtering  
            - **Neural Network**: Deep learning recommendations
            """)
        
        # Visualization
        if params["n_components"] >= 2:
            st.subheader("🔍 Cluster Visualization")
            X_pca = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X))
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", alpha=0.6, s=50)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("PCA + KMeans Clusters (User Segmentation)")
            
            # Add legend for clusters
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.tab10(i/len(cluster_stats)), 
                                        markersize=8, label=f'Cluster {i}') 
                             for i in range(len(cluster_stats))]
            ax.legend(handles=legend_elements, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explain what the visualization shows
            with st.expander("ℹ️ Understanding the PCA Plot"):
                st.write("""
                - **Each point** represents a user in the dataset
                - **Colors** represent different clusters of users with similar preferences
                - **Principal Components 1 and 2** capture the most important patterns in user behavior
                - **Users closer together** have more similar course enrollment patterns
                - **Well-separated clusters** indicate distinct user segments with different preferences
                """)
        
        # Show some technical details
        with st.expander("🔧 Technical Details"):
            st.write(f"**PCA Components:** {params['n_components']}")
            st.write(f"**KMeans Clusters:** {params['n_clusters']}")
            st.write(f"**Popularity Threshold:** {params['pop_threshold']} enrollments")
            st.write(f"**Features Used:** {len(FEATURE_NAMES)} features")
            
            # Show explained variance for PCA
            if params["n_components"] >= 2:
                explained_variance = pipe.named_steps["pca"].explained_variance_ratio_
                st.write(f"**PCA Explained Variance:**")
                for i, variance in enumerate(explained_variance):
                    st.write(f"  - PC{i+1}: {variance:.1%}")
                st.write(f"**Total Variance Explained:** {sum(explained_variance):.1%}")
            
elif model_selection == backend.models[3]:
    # Button to request KNN execution
    if st.sidebar.button("▶️ Recommend with Item-KNN", key="run_knn_btn"):
        st.session_state.knn_state['run_requested'] = True
        st.session_state.knn_state['completed'] = False
        st.session_state.knn_state['df'] = None

    # If the run is requested and not completed, then run
    if st.session_state.knn_state['run_requested'] and not st.session_state.knn_state['completed']:
        # Get all user IDs from the updated ratings
        user_ids = list(backend.load_ratings()["user"].unique())
        # If we have a test user, use only that user for faster results
        if st.session_state.get('test_user_id') is not None:
            user_ids = [st.session_state.test_user_id]

        with st.spinner("Running item-based KNN (this may take a minute)…"):
            knn_df = backend.predict(model_selection, user_ids, params)
            st.session_state.knn_state['df'] = knn_df
            st.session_state.knn_state['completed'] = True
            st.session_state.knn_state['run_requested'] = False

    # Display results if available
    if st.session_state.knn_state['completed']:
        st.success("✅ KNN finished!")
        st.dataframe(st.session_state.knn_state['df'])
#elif model_selection == backend.models[5]:
 #   if st.sidebar.button("▶️ Train & Recommend (NMF)", key="run_nmf_btn"):
  #      if not st.session_state.get('test_user_id'):
   #         st.warning("Please create a test user first!")
    #    else:
     #       user_id = st.session_state.test_user_id
      #      with st.spinner("Training NMF model and generating recommendations..."):
                # Train model
       #         backend.train_nmf_model(params["n_factors"], params["n_epochs"])
                
                # Get recommendations
        #        recs = backend.nmf_recommendations(user_id, params["top_k"])
                
                # Prepare results
               #results = []
                #for cid, score in recs.items():
                 #   results.append({
                  #      'USER': user_id,
                   #     'COURSE_ID': cid,
                    #    'COURSE_TITLE': backend.get_title_map().get(cid, "Unknown Course"),
                     #   'SCORE': score
                    #})
                
                #df = pd.DataFrame(results)
                #st.subheader("📚 NMF Recommendations")
                #st.dataframe(df)

elif model_selection == backend.models[4]:
    if st.sidebar.button("▶️ Train & Recommend (Neural Network)", key="run_nn_btn"):
        if not st.session_state.get('test_user_id'):
            st.warning("Please create a test user first!")
        else:
            user_id = st.session_state.test_user_id
            try:
                # Ensure user is added to the system
                if st.session_state.user_courses:
                    backend.add_new_ratings(st.session_state.user_courses)
                
                with st.spinner("Training neural network (this may take a few minutes)..."):
                    # Train model and generate recommendations
                    df = backend.predict(
                        model_name=backend.models[5],
                        user_ids=[user_id],
                        params={
                            "embedding_size": params["embedding_size"],
                            "epochs": params["epochs"],
                            "top_k": params["top_k"]
                        }
                    )
                
                st.subheader("🧠 Neural Network Recommendations")
                st.dataframe(df)
                
            except Exception as e:
                st.error(f"Neural network error: {str(e)}")
                st.warning("Try these fixes:")
                st.markdown("1. Select at least 3 courses before creating test user")
                st.markdown("2. Click 'Add Ratings for New User' before training")
                st.markdown("3. Try with fewer training epochs (5-10)")
elif model_selection == backend.models[5]:  # Regression with Embeddings
    if st.sidebar.button("▶️ Recommend (Regression Embeddings)", key="run_reg_emb"):
        if not st.session_state.get('test_user_id'):
            st.warning("Please create a test user first!")
        else:
            user_id = st.session_state.test_user_id
            with st.spinner("Generating regression recommendations..."):
                df = backend.predict(
                    backend.models[6], 
                    [user_id], 
                    {"top_k": params["top_k"]}
                )
            st.subheader("📊 Regression with Embeddings Recommendations")
            st.dataframe(df)


# Classification with Embeddings prediction
elif model_selection == backend.models[6]:  # Classification with Embeddings
    if st.sidebar.button("▶️ Recommend (Classification Embeddings)", key="run_cls_emb"):
        if not st.session_state.get('test_user_id'):
            st.warning("Please create a test user first!")
        else:
            user_id = st.session_state.test_user_id
            with st.spinner("Generating classification recommendations..."):
                df = backend.predict(
                    backend.models[7], 
                    [user_id], 
                    {"top_k": params["top_k"]}
                )
            st.subheader("📈 Classification with Embeddings Recommendations")
            
            if not df.empty:
                # Display results with styling
                st.dataframe(df.style.format({"SCORE": "{:.3f}"}))
            else:
                st.warning("No recommendations found. Try selecting different courses.")
             
