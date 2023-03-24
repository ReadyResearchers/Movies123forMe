#!/bin/bash
# git clone https://github.com/solisa986/Movies123forMe.git
# cd Movies123forMe
git init
git add .
git commit -m "autoupdate `date +%F-%T`"
git push origin main --force