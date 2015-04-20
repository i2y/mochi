Sending a Pull Request
======================

Use the following checklist to make sure your pull request can be reviewed and merged as efficiently as possible:

- If you are sending the PR for code review only and *not* for merge, add the "[WIP]" prefix to the PR's title.
- Write a descriptive Pull Request description (See sample below). Ideally, it should communicate:
    - Purpose
    	- The overall philosophy behind the changes you've made, so that, if there are questions as to whether an implementation detail was appropriate, this can be consulted before talking with the developer.
    	- Which Github issue the PR addresses, if applicable.
    - Changes. 
    	- The details of the implementation as you intended them to be. If you did front-end changes, add screenshots here.
    - Side effects. 
    	- Potential concerns, esp. regarding security, privacy, and provenance, which will requires extra care during review.


Sample Pull Request Description
===============================

Purpose
-------

Currently, the only way to add projects to your Dashboard's Project Organizer is from within the project organizer. There are smart folders with your projects and registrations, and you can search for projects from within the info widget to add to folders, but if you are elsewhere in the OSF, it's a laborious process to get the project into the Organizer. This PR allows you to be on a project and add the current project to the Project Organizer.

Closes Issue https://github.com/CenterForOpenScience/openscienceframework.org/issues/1186

Changes
-------

Puts a button on the project header that adds the current project to the Dashboard folder of the user's Project Organizer. Disabled if the user is not logged in, if the folder is already in the dashboard folder of the user's project organizer, or if the user doesn't have permissions to view.

Side Effects
------------

Also fixes a minor issue with the watch button count being slow to update.



__note: documentation for Pull Requests adapted from http://cosdev.readthedocs.org/en/latest/process/pull_requests.html__
