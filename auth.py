# import streamlit as st
# import hashlib
# from db import create_usertable, add_userdata, login_user

# # Hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def verify_password(password, hashed_password):
#     return hash_password(password) == hashed_password

# # Signup Page
# def signup_page():
#     st.subheader("Create New Account")
#     new_user = st.text_input("Username")
#     new_password = st.text_input("Password", type='password')
#     confirm_password = st.text_input("Confirm Password", type='password')

#     if st.button("Sign Up"):
#         if new_user and new_password:
#             if new_password == confirm_password:
#                 create_usertable()
#                 add_userdata(new_user, hash_password(new_password))
#                 st.success("Account created successfully!")
#                 st.session_state['logged_in'] = True
#                 st.session_state['username'] = new_user
#                 st.experimental_rerun()  # ✅ Force rerun to redirect
#             else:
#                 st.error("Passwords do not match.")
#         else:
#             st.error("Please fill in all fields.")

# # Login Page
# def login_page():
#     st.subheader("Login to Your Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type='password')

#     if st.button("Login"):
#         if username and password:
#             result = login_user(username, hash_password(password))
#             if result:
#                 st.success(f"Welcome, {username}!")
#                 st.session_state['logged_in'] = True
#                 st.session_state['username'] = username
#                 st.experimental_rerun()  # ✅ Force rerun to redirect
#             else:
#                 st.error("Incorrect Username or Password.")
#         else:
#             st.error("Please enter both username and password.")










# import streamlit as st
# import json
# import os
# import hashlib

# USER_DB = "users.json"

# def load_users():
#     if not os.path.exists(USER_DB):
#         with open(USER_DB, 'w') as f:
#             json.dump({}, f)
#     with open(USER_DB, 'r') as f:
#         return json.load(f)

# def save_users(users):
#     with open(USER_DB, 'w') as f:
#         json.dump(users, f, indent=4)

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def signup_page():
#     st.title("📝 Create Account")

#     with st.form("signup_form"):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         confirm = st.text_input("Confirm Password", type="password")
#         submit = st.form_submit_button("Sign Up")

#         if submit:
#             users = load_users()

#             if username in users:
#                 st.error("❌ Username already exists.")
#             elif password != confirm:
#                 st.error("❌ Passwords do not match.")
#             elif len(password) < 5:
#                 st.error("❌ Password too short (min 5 characters).")
#             else:
#                 users[username] = hash_password(password)
#                 save_users(users)
#                 st.success("✅ Account created!")
#                 st.info("Redirecting to Sign In page...")
#                 st.session_state.page = "signin"
#                 st.experimental_rerun()

#     st.markdown("---")
#     if st.button("Already have an account? Sign In"):
#         st.session_state.page = "signin"
#         st.experimental_rerun()

# def login_page():
#     st.title("🔐 Sign In")

#     with st.form("login_form"):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         submit = st.form_submit_button("Login")

#         if submit:
#             users = load_users()
#             if username in users and users[username] == hash_password(password):
#                 st.session_state.logged_in = True
#                 st.session_state.username = username
#                 st.success("✅ Logged in successfully!")
#                 st.session_state.page = "home"
#                 st.experimental_rerun()
#             else:
#                 st.error("❌ Invalid username or password.")

#     st.markdown("---")
#     if st.button("Don't have an account? Sign Up"):
#         st.session_state.page = "signup"
#         st.experimental_rerun()














# auth.py
import streamlit as st
from db import verify_user, add_user
import hashlib

def login_page():
    """Handles the login page UI and logic"""
    st.title("🔐 Sign In")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Invalid username or password")
            except Exception as e:
                st.error(f"⚠️ Database error: {str(e)}")
    
    if st.button("Don't have an account? Sign Up"):
        st.session_state.page = "signup"
        st.experimental_rerun()

def signup_page():
    """Handles the signup page UI and logic"""
    st.title("📝 Create Account")
    
    with st.form("signup_form"):
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            try:
                if not username or not password:
                    st.error("❌ Username and password cannot be empty")
                elif password != confirm:
                    st.error("❌ Passwords don't match")
                elif len(password) < 5:
                    st.error("❌ Password too short (min 5 characters)")
                else:
                    if add_user(username, password):
                        st.success("✅ Account created! Please login.")
                        st.session_state.page = "signin"
                        st.experimental_rerun()
                    else:
                        st.error("❌ Username already exists")
            except Exception as e:
                st.error(f"⚠️ Database error: {str(e)}")
    
    if st.button("Already have an account? Sign In"):
        st.session_state.page = "signin"
        st.experimental_rerun()