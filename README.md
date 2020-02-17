# First steps with Word Embedding : link between a country and capital in an embedded space

The goal of this study is to show a spatial link between a country and its capital in the embedded space.
To do so, I first tried to reduce the embedded space to 2 dimensions with ACP, and plotted different countries and their capitals in order to see wether there was an obvious spatial relation.

Then, considering I was losing too much information by reducing the embedded space to 2 dimensions, I decided to keep the 6 first dimensions given by the ACP.
Then, I computed each country-capital vector and created a matrix representing the cosine similarity between each vector. Indeed, if the same relation exists between each country and its capital, then all the matrix's elements should be very close to one.

The last part of the code contains my attempts to represent this matrix in a visual way.