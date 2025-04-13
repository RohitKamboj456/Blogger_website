import os
import datetime
import uuid
from flask import Flask, render_template_string, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey123!@#'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# -------------------- Models --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    avatar = db.Column(db.String(200), default='default-avatar.png')
    is_admin = db.Column(db.Boolean, default=False)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    comments = db.relationship('Comment', backref='post', lazy=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'))
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'))
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]))

# -------------------- ML Recommendation --------------------
class BlogRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.posts = []
        
    def train(self):
        self.posts = Post.query.all()
        texts = [post.content for post in self.posts]
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def get_similar_posts(self, post_id, num=3):
        if not hasattr(self, 'tfidf_matrix'):
            return []
        try:
            post_idx = next(i for i, post in enumerate(self.posts) if post.id == post_id)
            cosine_sim = cosine_similarity(self.tfidf_matrix[post_idx], self.tfidf_matrix)
            similar_indices = cosine_sim.argsort()[0][-num-1:-1][::-1]
            return [self.posts[i] for i in similar_indices]
        except Exception:
            return []

recommender = BlogRecommender()

# -------------------- File Upload Utilities --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4()}_{filename}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_name))
        return unique_name
    return None

# -------------------- Flask-Login User Loader --------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- Base HTML Template --------------------
base_html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tech Blog</title>
    <style>
        :root {
            --pink: #FF69B4;
            --green: #00FF00;
            --yellow: #FFFF00;
            --orange: #FFA500;
            --purple: #800080;
            --gradient: linear-gradient(45deg, var(--pink), var(--purple));
        }
        @keyframes slideIn {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .post-card {
            animation: slideIn 0.6s ease-out;
            transition: all 0.3s;
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .post-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        nav {
            background: var(--gradient);
            padding: 15px;
            margin-bottom: 30px;
        }
        nav a {
            color: white;
            margin: 0 15px;
            text-decoration: none;
            font-weight: bold;
        }
        button {
            background: var(--purple);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            transform: scale(1.05);
            background: var(--pink);
        }
        .flash {
            background: var(--green);
            color: black;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        {% if current_user.is_authenticated %}
            <a href="/create">New Post</a>
            <a href="/logout">Logout</a>
        {% else %}
            <a href="/login">Login</a>
            <a href="/register">Register</a>
        {% endif %}
    </nav>
    <div class="container">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="flashes">
                    {% for message in messages %}
                        <div class="flash">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        <!-- CONTENT_PLACEHOLDER -->
    </div>
    <script>
        // Simple comment reply toggle
        document.querySelectorAll('.reply-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const formId = btn.dataset.target;
                document.getElementById(formId).classList.toggle('hidden');
            });
        });
    </script>
</body>
</html>
'''

# -------------------- Helper Function to Render Templates --------------------
def render_with_base(content_template, **context):
    full_template = base_html.replace('<!-- CONTENT_PLACEHOLDER -->', content_template)
    return render_template_string(full_template, **context)

# -------------------- Routes --------------------
@app.route('/')
def home():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    content = '''
    <div class="posts">
        {% for post in posts %}
        <article class="post-card">
            {% if post.image %}
            <img src="{{ url_for('uploaded_file', filename=post.image) }}" alt="{{ post.title }}" style="max-width: 300px;">
            {% endif %}
            <h2 style="color: var(--purple);">{{ post.title }}</h2>
            <p>{{ post.content[:200] }}...</p>
            <a href="/post/{{ post.id }}" style="color: var(--green);">Read More →</a>
            {% if current_user.is_authenticated and current_user.is_admin %}
            <form action="/delete_post/{{ post.id }}" method="POST">
                <button type="submit" style="background: var(--orange);">Delete</button>
            </form>
            {% endif %}
        </article>
        {% endfor %}
    </div>
    '''
    return render_with_base(content, posts=posts)

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def view_post(post_id):
    post = Post.query.get_or_404(post_id)
    comments = Comment.query.filter_by(post_id=post_id, parent_id=None).order_by(Comment.created_at.desc()).all()
    similar_posts = recommender.get_similar_posts(post.id)
    if request.method == 'POST' and current_user.is_authenticated:
        comment_content = request.form.get('content')
        parent_id = request.form.get('parent_id')
        new_comment = Comment(
            content=comment_content,
            post_id=post_id,
            user_id=current_user.id,
            parent_id=parent_id if parent_id else None
        )
        db.session.add(new_comment)
        db.session.commit()
        return redirect(url_for('view_post', post_id=post_id))
    content = '''
    <article class="post-card">
        <h1 style="color: var(--pink);">{{ post.title }}</h1>
        {% if post.image %}
        <img src="{{ url_for('uploaded_file', filename=post.image) }}" alt="{{ post.title }}" style="max-width: 100%;">
        {% endif %}
        <div class="post-content">{{ post.content }}</div>
        <div class="comments" style="margin-top: 30px;">
            <h3 style="color: var(--green);">Comments ({{ comments|length }})</h3>
            {% for comment in comments %}
                <div class="comment" style="margin: 15px 0; padding: 10px; border-left: 3px solid var(--yellow);">
                    <p>{{ comment.content }}</p>
                    <button class="reply-btn" data-target="reply-{{ comment.id }}">Reply</button>
                    <form id="reply-{{ comment.id }}" class="hidden" method="POST" style="margin-top: 10px;">
                        <textarea name="content" required style="width: 100%;"></textarea>
                        <input type="hidden" name="parent_id" value="{{ comment.id }}">
                        <button type="submit">Post Reply</button>
                    </form>
                </div>
            {% endfor %}
        </div>
        {% if current_user.is_authenticated %}
        <form method="POST" style="margin-top: 20px;">
            <textarea name="content" required style="width: 100%; height: 100px;"></textarea>
            <button type="submit">Add Comment</button>
        </form>
        {% endif %}
        <div class="similar-posts" style="margin-top: 30px;">
            <h3 style="color: var(--orange);">Recommended Posts</h3>
            {% for post in similar_posts %}
                <a href="/post/{{ post.id }}" style="display: block; color: var(--purple);">{{ post.title }}</a>
            {% endfor %}
        </div>
    </article>
    '''
    return render_with_base(content, post=post, comments=comments, similar_posts=similar_posts)

@app.route('/create', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        image = request.files.get('image')
        filename = save_file(image) if image else None
        new_post = Post(
            title=title,
            content=content,
            image=filename,
            author_id=current_user.id
        )
        db.session.add(new_post)
        db.session.commit()
        recommender.train()
        return redirect(url_for('home'))
    content = '''
    <div class="post-form" style="max-width: 800px; margin: 0 auto;">
        <form method="POST" enctype="multipart/form-data">
            <input type="text" name="title" placeholder="Post Title" required style="width: 100%; padding: 10px;">
            <textarea name="content" placeholder="Write your post..." required 
                      style="width: 100%; height: 300px; padding: 10px;"></textarea>
            <input type="file" name="image" style="margin: 10px 0;">
            <button type="submit">Publish 🚀</button>
        </form>
    </div>
    '''
    return render_with_base(content)

@app.route('/delete_post/<int:post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if current_user.is_admin or post.author_id == current_user.id:
        db.session.delete(post)
        db.session.commit()
        flash('Post deleted successfully')
    return redirect(url_for('home'))

# -------------------- Authentication Routes --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials')
    content = '''
    <div style="max-width: 400px; margin: 0 auto;">
        <form method="POST">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
    </div>
    '''
    return render_with_base(content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    content = '''
    <div style="max-width: 400px; margin: 0 auto;">
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Register</button>
        </form>
    </div>
    '''
    return render_with_base(content)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------- Initialize Database and Admin User --------------------
with app.app_context():
    db.create_all()
    recommender.train()
    if not User.query.filter_by(is_admin=True).first():
        admin = User(
            username='admin',
            email='admin@blog.com',
            password=generate_password_hash('Admin123!'),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 