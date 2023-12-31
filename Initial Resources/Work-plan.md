## Using Machine Learning to predict the first visibility of the New Crescent Moon in the UK

The visual sighting of the new crescent Moon is a key aspect of many religious and cultural calendars. Although the phase of the Moon can be precisely calculated, its actual observability as a thin crescent can depend on a number of factors. Previous academic studies have been based on data recorded at latitudes much lower than that of the UK, in locations where the weather is more astronomically favourable.

The aim of this project would be to combine UK observation reports of the crescent Moon, with other factors such as archival weather data, and build a machine learning algorithm to predict the likelihood of a lunar crescent being visible in the UK for any given month. The students will gain familiarity with astronomical Python libraries and their use with luni-solar ephemeris data, in addition to the development and application of machine learning algorithms to varied real-world datasets.

This project will be particularly well suited for students with interests in machine learning, the practicalities of amateur astronomy, and the cultural impact of astronomy. Some prior knowledge and experience of programming in Python would be beneficial.

---

### Work Plan

1. Literature review covering astrophysics (e.g. luni-solar geometries; criteria for new crescent visibility e.g. Yallop/Odeh; observational practicalities) and machine learning (e.g. overview of use of ML in astrophysics, determination of most appropriate algorithm(s) for this project). 
2. Application of mathematics in Yallop 1997 (https://www.astronomycenter.net/pdf/yallop_1997.pdf) to reproduce q value calculations and visibility maps. 
3. Processing of archival Moon visibility data to form a ML-readable dataset, including augmenting the data (calculate useful parameters, add other relevent data etc).
4. Testing of different ML algorithms to evaluate their usefulness in this application, and apply the best to make future predictions. Evaluate what information is most important to make such predictions. 
5. Attending weekly meetings with Prof. Anna Scaife's ML group (times and locations TBC). Share updates of your work (approx 2 slides weekly) and discuss project progression.
6. Interpret and discuss results.

Optional extras: 

a. Additional section in literature review on cultural impact of the Moon/ astronomy in general (no more than 20\% of total lit review content).
b. Evaluation of ways to present first visibility maps to allow for greatest accessibility, especially for months which have a large public interest (e.g. the Islamic month of Ramadan). Creation of resources to aid in public understanding of the lunar cycle and how to observe it. (No more than 10\% of total project time)

---

### Getting started 

1. Get setup with accounts on the JBCA computing system and the Galahad cluster. I will contact Dr Ant Holloway and/or Dr Sotirios Sanidas for this; Anna may also be coordinating for other MPhys students in her group. There is a JBCA-ICE Slack network to join. 
2. Gain familiarity with running scripts on Galahad (linux environment). To begin with, Jupyter notebooks may be the easiest to use. A guide to running Jupyter on Galahad can be found [here](https://github.com/zhaotingchen/galahad_guide/blob/main/jupyter_on_galahad.md). 
3. Conduct a literature review as described in the overall Work Plan. This is likely to take up at least the first couple of MPhys workdays (alongside computer setup — you may have to wait for logins to activate etc, so do this in parallel). Think of it as writing an early draft of your report's introduction section. Of course, as you progress through the project, different things may become more or less relevant.
4. Dive into the world of ML with Python — lots of online resources for this! Lean about different types of ML algorithms (e.g. supervised vs unsupervised) and how they can suit different sorts of data.

---

### Possible lines of investigation

1. Weather: cloud cover (percentage and direction), other atmospheric effects.
2. Elevation (i.e. height obeservation is made from).
3. Capabilities of human eye (resolution, dynamic range etc).
4. Something else...?

----

#### Useful Links

Use these to get an initial introduction to the project, and as a starting point for your literature review. 

* [Webpages of RH van Gent](https://webspace.science.uu.nl/~gent0113/islam/islam_lunvis.htm), particually [this extensive list](https://webspace.science.uu.nl/~gent0113/islam/islam_lunvis_references.htm) of existing literature. 
* [Yallop 1997 “A method for predicting the first sighting of the new Crescent Moon”](https://www.astronomycenter.net/pdf/yallop_1997.pdf)
* [Hoffman 2003 “Observing the new Moon”](https://academic.oup.com/mnras/article/340/3/1039/1746574)
* [Hafez et al 2014 “A radio determination of the time of the New Moon”](https://academic.oup.com/mnras/article/439/3/2271/1084163)
* [Al-Rajab et al 2023 “Predicting new crescent moon visibility applying machine learning algorithms”](https://www.nature.com/articles/s41598-023-32807-x)
