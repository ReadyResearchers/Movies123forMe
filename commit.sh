#!/bin/bash
git init
git add .
git commit -m "autoupdate `date +%F-%T`"
git push https://github.com/solisa986/Movies123forMe.git main --force