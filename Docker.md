Using Docker
============

## How to use Docker image
- Install Docker for your appropriate environment, see [Docker
  installation documentation] (https://docs.docker.com/installation)
- Mochi Docker image usage is explained on its [Docker hub]
  (https://registry.hub.docker.com/u/tlvu/mochi)

## Why Docker
- Full-stack (OS, python libs), reproducible, easy to distribute runtime
  environment
  - The exact OS, all system packages versions, all python libraries
    versions are all pinned down.
  - Therefore the exact same runtime environment is reproducible on every
    developper and user machine.
  - The exact same Docker image works on Linux and Mac and is
    completely self-contained (all dependencies are part of the image, user
    has nothing else to install).
  - Provide the "self-contained" advantage of a full virtual machine (ex:
    VirtualBox, VMWare) without the performance and heavy disk space usage
    penalty.
```sh
$ time docker run -i -t --rm -v $MOCHI_CHECKOUT_ROOT:/files:ro tlvu/mochi /files/examples/aif.mochi
empty
10

real    0m6.612s
user    0m0.006s
sys     0m0.017s

$ docker images docker.io/tlvu/mochi
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
docker.io/tlvu/mochi   0.2.1               4a44168346a3        About an hour ago   426.8 MB
docker.io/tlvu/mochi   0.2.1-20150415      4a44168346a3        About an hour ago   426.8 MB
docker.io/tlvu/mochi   latest              4a44168346a3        About an hour ago   426.8 MB
```

## Releasing a new Docker image
- Build a new image
```sh
$ cd $MOCHI_CHECKOUT_ROOT
$ docker build -t mochi .
```

- Tag the new build appropriately with the new version.  Example from 0.2.1
  release:
```sh
$ docker tag -f mochi tlvu/mochi:0.2.1  # -f to force move existing tag if needed

# This tag should be unique for each release (should never use '-f' here).  If
# releasing more than once in same day, append something to the date to make it
# different, ex: 0.2.1-20150415-2.
$ docker tag mochi tlvu/mochi:0.2.1-20150415

# Need 'latest' tag for 'tlvu/mochi<no tags specified>' to work.  This tag is
# sure to already exist so we need '-f'.
$ docker tag -f mochi tlvu/mochi:latest
```

- Push the image to Docker hub
```sh
$ docker push tlvu/mochi
```
