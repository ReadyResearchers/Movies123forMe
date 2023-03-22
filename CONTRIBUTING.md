## Contributing

First off, thank you for considering contributing to Active Admin. It's people
like you that make Active Admin such a great tool.

### Where do I go from here?

If you've noticed a bug or have a feature request, [make one][new issue]! It's
generally best if you get confirmation of your bug or approval for your feature
request this way before starting to code.

If you have a general question about activeadmin, you can post it on [Stack
Overflow], the issue tracker is only for bugs and feature requests.

### Fork & create a branch

If this is something you think you can fix, then [fork Active Admin] and create
a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-japanese-translations
```

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help;
everyone is a beginner at first :smile_cat:

### View your changes in a Streamlit application

Movies123forMe is meant to be used by humans, not cucumbers. So make sure to take
a look at your changes in a browser.

Simply run the commands provided in the README.md to connect your application instance to Streamlit locally and then navigate to http://localhost:8501/
to see your changes!

### Get the style right

Your patch should follow the same conventions & pass the same code quality
checks as the rest of the project. `bin/rake lint` will give you feedback in
this regard. You can check & fix style issues by running each linter
individually. Run `bin/rake -T lint` to see the available linters.

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's
up to date with Active Admin's master branch:

```sh
git remote add upstream git@github.com:activeadmin/activeadmin.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-japanese-translations
git rebase master
git push --set-upstream origin 325-add-japanese-translations
```

Finally, go to GitHub and [make a Pull Request][] :D

Github Actions will run our test suite against all supported Rails versions. We
care about quality, so your PR won't be merged until all tests pass. It's
unlikely, but it's possible that your changes pass tests in one Rails version
but fail in another. In that case, you'll have to setup your development
environment (as explained in step 3) to use the problematic Rails version, and
investigate what's going on!

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code
has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing in Git, there are a lot of [good][git rebasing]
[resources][interactive rebase] but here's the suggested workflow:

```sh
git checkout 325-add-japanese-translations
git pull --rebase upstream master
git push --force-with-lease 325-add-japanese-translations
```

### Merging a PR (maintainers only)

A PR can only be merged into master by a maintainer if:

* It is passing CI.
* It has been approved by at least two maintainers. If it was a maintainer who
  opened the PR, only one extra approval is needed.
* It has no requested changes.
* It is up to date with current master.

Any maintainer is allowed to merge a PR if all of these conditions are
met.

### Shipping a release (maintainers only)

Maintainers need to do the following to push out a release:

* Switch to the master branch and make sure it's up to date.
* Make sure you have [chandler] properly configured. Chandler is used to
  automatically submit github release notes from the changelog right after
  pushing the gem to rubygems.
* Run one of `bin/rake release:prepare_{prerelease,prepatch,patch,preminor,minor,premajor,major}`, push the result and create a PR.
* Review and merge the PR. The generated changelog in the PR should include all user visible changes you intend to ship.
* Run `bin/rake release` from the target branch once the PR is merged.
