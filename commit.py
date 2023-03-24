from git import Repo

PATH_OF_GIT_REPO = 'git@github.com:solisa986/Movies123forMe.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = "autoupdate `date +%F-%T`"

def git_push():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')    

git_push()