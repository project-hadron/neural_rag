# Difference Component Service
Taking two input datasets, produces a set of data profiling for each, then given a unique identifier common 
to both, compares the difference between each dataset where the identifiers match. The resulting output shows
the rows and columns where value difference is found. 

### General Environment Variables
Create the environment variables
* HADRON_DOMAIN_REPO_PATH - the location path of the Hadron Controller and its component capabilities
* HADRON_DEFAULT_PATH - a default path where to put non-specified outputs.

### Component Specific Environment Variables
The following environment variables are specific to this Difference Report
* 

## Hadron docker-compose example
 an example yaml file might look like

```yaml
version: '3.8'
services:
  domain-controller:
    image: gigas64/project_hadron:3.4
    env_file: private_env_file
    environment:
      # mandatory domain ensemble path
      - HADRON_DOMAIN_REPO_PATH=
      # controller startup envs (optional)
      - HADRON_CONTROLLER_REPORT=
      # Connector contract paths (optional)
      - HADRON_DEFAULT_PATH=
      # specific component envs

```
where the `private_env_file` contains private environment variables such as connector secrets or tokens and
the `gigas64/project_hadron:2.1` image is the generic docker image to run Project Hadron Component Services.

If you are not using thr `private_env_file` then remove the `env_file:` entry

Then to start the docker container, from the location of your `docker-compose.yml` file, run
```bash
$> docker-compose up -d
```

To stop the docker container, run
```bash
$> docker-compose down
$> docker-compose stop
```

To find the networks, run
```bash
$> docker network ls
```
