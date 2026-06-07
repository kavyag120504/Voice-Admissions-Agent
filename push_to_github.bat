@echo off
cd /d "D:\NLP PROJECT (2)\NLP PROJECT\bmu-advanced-call-agent"

git init
git branch -m main
git add .
git status
git commit -m "Initial commit: BMU Voice Admissions Agent - Real-time multilingual NLP chatbot"
git remote add origin https://github.com/kavyag120504/Voice-Admissions-Agent.git
git push -u origin main

echo Done! Check GitHub.
pause
