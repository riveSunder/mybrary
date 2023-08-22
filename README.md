# mybrary: CLI semantic search for individual PDFs

`mybrary` is a simple tool I use to search pdfs using vector similarity metrics on tranformer language model embeddings. 

# Getting started

I use `virtualenv` to manage dependencies in a virtual environment:

```
git clone git@github.com:riveSunder/mybrary.git
# git clone https://github.com/riveSunder/mybrary.git

cd mybrary

virtualenv mybrary_env --python=python3.8
source mybrary_env/bin/activate

pip install pdfminer.six==20221105 torch transformers
# or 
# pip install -r requirements.txt
```

# Usage

With the `brary/pdf_search.py` you can make a semantic search query of a single pdf from the command line. You can check the available input arguments by calling help with `python -m brary.pdf_search -h


```
usage: pdf_search.py [-h] [-q QUERY [QUERY ...]] [-d DIRECTORY] [-i INPUT] [-k K]

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY [QUERY ...], --query QUERY [QUERY ...]
                        query or queries to search for in pdf text(separate by space and use single quote marks)
  -d DIRECTORY, --directory DIRECTORY
  -i INPUT, --input INPUT
                        filename for pdf of interest
  -k K, --k K           k for top-k matches to report
```

You can make multiple queries enclosed in single or double quotation marks to distinguish each string. Passing an integer value of 2 or greater to `-k` will return the top-k results, and the tool returns results for cosine similarity and L2 distance separately by default.


As an example, we can query the book [Conway's Game of Life: Mathematics and Construction](https://conwaylife.com/book/) by Nathaniel Johnston and Dave Greene with a statement about gliders and universal computation (Thanks to the authors for making a free pdf available, -> [the hardcover](https://www.lulu.com/shop/dave-greene-and-nathaniel-johnston/conways-game-of-life/hardcover/product-ev72jn.html) is available for purchase). 

The query:

```
python -m brary.pdf_search -q "Gliders, like the reflex glider discovered by Richard Guy in the 60s, and other mobile patterns provide the basic functions for building universal computation: moving, transforming (in their collisions), and storing (via collision synthesis/destruction) information." -i conway_life_book.pdf -k 1
```

yields a related snippet:

``` 
loading pre-computed vectors from data/conway_life_book.pt

0th best match for query Gliders, like the reflex glider discovered by Richard Guy in the 60s, and other mobile patterns provide the basic functions for building universal computation: moving, transforming (in their collisions), and storing (via collision synthesis/destruction) information.
	 with cosine similarity 0.721

 Remarks

The importance of glider synthesis was known essentially as soon as the glider itself was found in
1970, with common folklore being that we could send gliders as signals throughout the Life plane
and collide those gliders in different ways to simulate arbitrary computations. This basic idea has
been reﬁned and made more precise repeatedly over the past 50 years, to the point that there are now
explicit patterns that do exactly this—they collide gliders so as to perform arbitrary computations and
build almost any pattern of our choosing (we will delve deeply into the speciﬁcs of how these patterns
work in
0th best match for query Gliders, like the reflex glider discovered by Richard Guy in the 60s, and other mobile patterns provide the basic functions for building universal computation: moving, transforming (in their collisions), and storing (via collision synthesis/destruction) information.
	 with l2 distance 4.438

 Remarks

The importance of glider synthesis was known essentially as soon as the glider itself was found in
1970, with common folklore being that we could send gliders as signals throughout the Life plane
and collide those gliders in different ways to simulate arbitrary computations. This basic idea has
been reﬁned and made more precise repeatedly over the past 50 years, to the point that there are now
explicit patterns that do exactly this—they collide gliders so as to perform arbitrary computations and
build almost any pattern of our choosing (we will delve deeply into the speciﬁcs of how these patterns
work in
enter another query (0 to end)

```

In the above case cosine similarity and l2 distance return the same passage, but these sometimes differ. 
