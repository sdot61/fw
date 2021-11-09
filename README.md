# fw

## Deployment

This App is setup for auto deployment via Github Actions and AWS CodePipeline and CodeDeploy
Triggering new deployments is done by commiting code changes to branch `master` on this github repo

This can be done from the commandline:
```
    # navigate to project...
    $ cd /path/to/project
    
    # add changes ...
    $ git add .
    
    # stage the commit
    $ git commit -m "I have reasons for this commit"
    
    # push to github...
    $ git push origin master
```

A git client configured to access this github repo can work as well. Once the commit has been pushed to Github,
the deployment should show up in the CodeDeploy and CodePipeline consoles. Once the deployment completes, the 
committed changes will be live

## Launching new Lightsail Instances
- copy the contents of `.lightsail-launchscript.sh` from the private `fw-code-deploy` bucket on S3
- in the Lightsail console, click "Create Instance"
- select "Linux" and "OS Only"
- under "optional" click "add launch script"
- paste the contents of `.lightsail-launchscript.sh` into the text area
- under "Identify your instance" where it says "Amazon_Linux_2-1" etc, change that to a unique name
- under Key-value tags click "Add"
- enter key "GROUP" with value "codedeploy"
- click "Create instance" button
