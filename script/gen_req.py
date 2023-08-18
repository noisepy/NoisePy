# quick and dirty script to extract package dependencies
# used by docker_dev.sh
import sys

import pkginfo

wheel = sys.argv[1]
reqs = pkginfo.get_metadata(wheel).requires_dist
core_reqs = [req for req in reqs if "extra" not in req]
for req in core_reqs:
    print(req)
