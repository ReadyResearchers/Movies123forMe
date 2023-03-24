#!/bin/bash
git add .
git commit -m "autoupdate `date +%F-%T`"
git push origin main --force