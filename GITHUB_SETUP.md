# GitHub Repository Setup Guide

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `EE599-project` (or your preferred name)
   - **Description**: "Low-Perplexity Masking for Catastrophic Forgetting Mitigation"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Run these commands in your terminal:

```bash
cd /home/exouser/Desktop/vscode/EE599-project

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/EE599-project.git

# Or if you prefer SSH (requires SSH key setup):
# git remote add origin git@github.com:YOUR_USERNAME/EE599-project.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

1. Go to your GitHub repository page
2. You should see all your files there
3. The README.md should be displayed on the main page

## Additional Commands

### Check remote repository
```bash
git remote -v
```

### Update remote URL (if needed)
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/EE599-project.git
```

### Push future changes
```bash
git add .
git commit -m "Your commit message"
git push
```

### Pull changes from GitHub
```bash
git pull origin main
```

## Notes

- If you haven't set up Git credentials, GitHub may prompt you for authentication
- For HTTPS, you may need to use a Personal Access Token instead of password
- For SSH, make sure you have SSH keys set up with GitHub

