# MapD Release Guide

Based on the [Docker Release Checklist](https://github.com/docker/docker/blob/master/project/RELEASE-CHECKLIST.md).

## General layout

- development occurs in master
- no dedicated `release` branch - only `release` tags

## Variables

| Variable | Description  | Format   | Example |
|  ---     |  ---         |  ---     |  ---    |
| `$VERS`  | new version  | vX.Y.Z   | v1.1.5  |
| `$LVERS` | last version | vX.Y.Z-1 | v1.1.4  |
| `$NVERS` | next version | vX.Y.Z+1 | v1.1.6  |

Versions are of the form `vX.Y.Z`, eg `v1.1.3`.

## Freeze, branch off `origin/master`

Branch name: `rc/$VERS`

    git checkout master
    git reset --hard origin/master
    git pull origin/master
    git branch rc/$VERS

## Bump version in `master`

Append `dev` to version: if we're releasing `v1.1.5`, bump version in `master` to `v1.1.6dev`.

Versions are currently managed in `CMakeLists.txt`.

Commit message:

    Bump to $NVERSdev

## Update version, commit

Under the `rc` branch, update version and frontend URL in `CMakeLists.txt`. Frontend URL should be a direct link to the specific frontend being bundled - no `latest` symlinks. Note: do not use HTTPS link. Some CMakes aren't built with https support.

Commit message:

    Bump to $VERS

## Gather changes

    git fetch --all --tags -v
    git log $LVERS..rc/$VERS

Or on GitHub:

    https://github.com/map-d/mapd2/compare/$LVERS...rc/$VERS

## Update `RELEASENOTES.md`, commit, iterate

Pare down commit log to meaningful messages for customers. Iterate by adding more commits (we'll squash them later).

Commit message:

    docs: init release notes, $VERS

## Submit RC to Jenkins

Submit branch to `mapd2-multi` project on Jenkins, specifying branch `rc/$VERS`.

TBD: update symlinks. Will eventually add as a checkbox in `mapd2-multi`.

## Deploy to test server(s)

TBD

## Squash release notes

Squash release notes commits.

Commit message:

    docs: finalize release notes, $VERS

## Tag version, push to origin

    git tag -a $VERS -m $VERS

(verify no bad tags)

    git push origin --tags

## Cherry-pick release notes to master

Grab hash for release notes commit, `$RNHASH`.

    git checkout master
    git pull
    git cherry-pick -x $RNHASH
    git push

## Submit final to Jenkins

Submit tag to `mapd2-multi` project on Jenkins, specifying branch `$VERS`.

TBD: update symlinks. Will eventually add as a checkbox in `mapd2-multi`.

## Delete RC branch on GitHub

## Handling hotfixes
Commit to `master` (if possible), cherry-pick over to `rc/$VERS`.

After cherry-pick, interactive rebase to move the release notes and version bump commits to the top.
