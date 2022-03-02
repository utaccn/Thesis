# Smooth solution space exploration by introducing local similarity search in a MAP-Elites algorithm

### Abstract

MAP-Elites algorithms have been developed and applied with the aim of better illuminating the search space of an evolutionary
algorithm by defining an archive of many niches within a behaviour space, where only individuals within a niche compete. Resulting in an
archive that illuminates possible solutions, with each niche containing the best performing prototypes found or an area of behaviour space.
The space of solutions, however, is not “smooth”, i.e., neighbouring niches in the behaviour space of the archive may contain solutions with
very different phenotypical traits. The aim of this research was to create an interactive tool to generate and explore archives produced by
the MAP-Elites algorithms in a visual context. Smooth archives support interaction by arranging similar phenotypes in proximity to each
other, allowing users to more easily navigate them. In order to do so, we have introduced measures of similarity among neighbouring niches
and compared the “smoothness” of our models with respect to the standard one. The comparison is presented for two different visual tasks.
The results demonstrate that our model, in the proper conditions, keeps the archive diversity and performance of the standard MAP-Elites
algorithm, but is able to additionally select the archive prototypes, generating a smoother archive.

---

Collection of the scripts used for my Master's Thesis project. 
You can find the original paper in this repository.

The code is based on the original MAP-Elites Python implementation [py_mapelites](https://github.com/resibots/pymap_elites)
