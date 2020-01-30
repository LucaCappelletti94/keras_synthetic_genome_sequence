keras_synthetic_genome_sequence
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability| |pip| |downloads|

Python package to lazily generate synthetic genomic sequences for training of Keras models.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install keras_synthetic_genome_sequence

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime
get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|


Usage examples
-------------------------
To use GapSequence to train your keras model you
will need to obtain statistical metrics for the
biological gaps you intend to mimic in your synthetic gaps.

To achieve this, this package offers an utility called
get_gaps_statistics, which allows you to obtain the
mean and covariance of gaps in a given genomic assembly.

The genomic assembly is automatically downloaded from UCSC
using `ucsc_genomes_downloader <https://github.com/LucaCappelletti94/ucsc_genomes_downloader>`__,
then the gaps contained within are extracted and their windows
is expanded to the given one, after filtering for the given
max_gap_size, as you might want to limit the gaps size to
a relatively small one (gaps can get in the tens of thousands
of nucleotides, for instance in the telomeres).

Let's start by listing all the important parameters:

.. code:: python

    assembly = "hg19"
    window_size = 200
    batch_size = 128

Now we can start by retrieving the gaps statistics:

.. code:: python

    from keras_synthetic_genome_sequence.utils import get_gaps_statistics

    number, mean, covariance = get_gaps_statistics(
        assembly=assembly,
        max_gap_size=100,
        window_size=window_size
    )

    print("I have identified {number} gaps!".format(number=number))

Now you must choose the ground truth on which to apply the
synthetic gaps, for instance the regions without gaps in
the genomic assembly hg19, chromosome chr1.
These regions will have to be tasselized into smaller
chunks that are compatible with the shape you have chosen for
the gap statistics window_size.
We can retrieve these regions as follows:

.. code:: python

    from ucsc_genomes_downloader import Genome
    from ucsc_genomes_downloader.utils import tasselize_bed

    hg19 = Genome(assembly, chromosomes=["chr1"])
    ground_truth = tasselize_bed(genome.filled(), window_size=window_size)

The obtained pandas DataFrame will have a bed-like format
and look as follows:

+----+---------+--------------+------------+
|    | chrom   |   chromStart |   chromEnd |
+====+=========+==============+============+
|  0 | chr1    |        10000 |      10200 |
+----+---------+--------------+------------+
|  1 | chr1    |        10200 |      10400 |
+----+---------+--------------+------------+
|  2 | chr1    |        10400 |      10600 |
+----+---------+--------------+------------+
|  3 | chr1    |        10600 |      10800 |
+----+---------+--------------+------------+
|  4 | chr1    |        10800 |      11000 |
+----+---------+--------------+------------+

Now we are ready to actually create the GapSequence:

.. code:: python

    from keras_synthetic_genome_sequence import GapSequence

    gap_sequence = GapSequence(
        assembly=assembly,
        ground_truth,
        gaps_mean=mean,
        gaps_covariance=covariance,
        batch_size=batch_size
    )

Now, having a model that receives as
input and output shape (batch_size, window_size, 4),
we can train it as follows:

.. code:: python

    model = build_my_denoiser()
    model.fit_generator(
        gap_sequence,
        steps_per_epoch=gap_sequence.steps_per_epoch,
        epochs=2,
        shuffle=True
    )

Happy denoising!

Comparison between biological and synthetic distributions
----------------------------------------------------------
The following images refer to the biological and synthetic distributions
of gaps in the hg19, hg38, mm9 and mm10 genomic assembly, considering
gaps with length to up 100 nucleotides and total window size 1000.
The threshold used to convert to integer the multivariate gaussian distribution
is 0.4, the default value used within the python package.

.. image:: https://github.com/LucaCappelletti94/keras_synthetic_genome_sequence/blob/master/distributions/hg19.png?raw=true
.. image:: https://github.com/LucaCappelletti94/keras_synthetic_genome_sequence/blob/master/distributions/hg38.png?raw=true
.. image:: https://github.com/LucaCappelletti94/keras_synthetic_genome_sequence/blob/master/distributions/mm9.png?raw=true
.. image:: https://github.com/LucaCappelletti94/keras_synthetic_genome_sequence/blob/master/distributions/mm10.png?raw=true


.. |travis| image:: https://travis-ci.org/LucaCappelletti94/keras_synthetic_genome_sequence.png
   :target: https://travis-ci.org/LucaCappelletti94/keras_synthetic_genome_sequence
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_synthetic_genome_sequence&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_synthetic_genome_sequence
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_synthetic_genome_sequence&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_synthetic_genome_sequence
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_synthetic_genome_sequence&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_synthetic_genome_sequence
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/keras_synthetic_genome_sequence/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/keras_synthetic_genome_sequence?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/keras_synthetic_genome_sequence.svg
    :target: https://badge.fury.io/py/keras_synthetic_genome_sequence
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/keras_synthetic_genome_sequence
    :target: https://pepy.tech/badge/keras_synthetic_genome_sequence
    :alt: Pypi total project downloads

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/7f2c4e2947834c05b5a869a9445482d0
    :target: https://www.codacy.com/manual/LucaCappelletti94/keras_synthetic_genome_sequence?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/keras_synthetic_genome_sequence&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/b89f6bd0ddc58cc93e89/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_synthetic_genome_sequence/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/b89f6bd0ddc58cc93e89/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_synthetic_genome_sequence/test_coverage
    :alt: Code Climate Coverate
