#!/usr/bin/env python3

import sys
import argparse
import csv
from geopy import distance
import numpy as np
import dateutil.parser
import datetime
from datetime import date

"""
   # Introduction

   This algorithm implements a simple species distribution model (SDM) based on nearest neighbors,
   using both spatial and temporal (seasonal) distances.

   Let `P` be the set of known points, each with an assigned species. For a given point `p` (lon,
   lat), the algorithm takes the `k` nearest points from `P` for each species. Let `Ks` be the
   set of `k` nearest points for species `s` with respect to `p`. Then, for each species `s`, the
   pairwise spatial distance between `p` and all points in `Ks` is computed and then, optionally,
   is weighted by the seasonal distance (difference between the two year days). Afterwards, the
   average, `Ms`, is taken.  Then, a score is derived based on the inverse of the distance:

              /  1  \
    score  = |  ---- |
         s    \  Ms /

   Finally, a probability (sum = 1) is computed by dividing each score by the their sum.


   # Input files

   The CSV file must have the following fields:

      species name,longitude,latitude[,date]

   Note: the field date is optional; when given it must be of the form yyyy-mm-dd


   # Usage
   Usage (example):

      ./sdm.py -c sample.csv -lon 52 -lat 18 -r 250


   # Parameter optimization (2020-07-09)

   Using a test CSV file containing 1000 SISS-Geo registers, the best parameter set is:

      -k2 -r175 -w0.3 -m1 -dw0.2 -> 51.2% accuracy

   The best parameters with -w0.0 (collaborator model disabled) is:

      -k2 -r175 -w0.0 -m1 -dw0.2 -> 49.6% accuracy,

   therefore, the collaborator model improves ~3% the accuracy. The best parameters with -dw0.0
   (disables seasonality) is:

      -k2 -r150 -w0.5 -m1 -dw0.0 -> 48.0% accuracy,

   thus, the seasonality weighting improves ~7% the accuracy.

   Regarding the number of neighbors (k parameter), using -k1 leads to 49.2% at best, while -k2
   results in 51.2% and -k3 in 50.1%:

   -k2 -r175 -w0.3 -m1 -dw0.2 -> 51.2% accuracy
   -k3 -r150 -w0.2 -m1 -dw0.2 -> 50.1% accuracy
   -k1 -r50  -w0.3 -m1 -dw0.3 -> 49.2% accuracy
   -k5 -r175 -w0.2 -m1 -dw0.2 -> 48.9% accuracy
   -k4 -r175 -w0.2 -m1 -dw0.2 -> 48.8% accuracy


   In terms of radius, we have:

   -k2 -r175 -w0.3 -m1 -dw0.2 -> 51.2% accuracy
   -k2 -r100 -w0.3 -m1 -dw0.3 -> 51.1% accuracy
   -k2 -r150 -w0.2 -m1 -dw0.2 -> 50.9% accuracy
   -k2 -r50  -w0.2 -m1 -dw0.4 -> 50.6% accuracy
   -k2 -r75  -w0.3 -m1 -dw0.3 -> 50.6% accuracy
   -k2 -r125 -w0.4 -m1 -dw0.2 -> 50.6% accuracy
   -k2 -r200 -w0.3 -m1 -dw0.2 -> 50.4% accuracy
   -k2 -r25  -w0.5 -m1 -dw0.6 -> 50.0% accuracy

   Finally, the minimum number of instances (-m), we have:

   -k2 -r175 -w0.3 -m1 -dw0.2 -> 51.2% accuracy
   -k2 -r100 -w0.3 -m2 -dw0.3 -> 50.8% accuracy
   -k2 -r100 -w0.2 -m3 -dw0.3 -> 50.3% accuracy
   -k2 -r100 -w0.3 -m4 -dw0.3 -> 50.3% accuracy
   -k2 -r175 -w0.4 -m5 -dw0.2 -> 49.9% accuracy

   which shows that it is better to always train the model (-m1).


   # Author
   Author: Douglas A. Augusto <daa@fiocruz.br>

   License: GNU GPL 3.0 Free Software
"""

def str_to_datetime(s):
   try:
      return dateutil.parser.parse(s)
   except:
      try:
         # Try to convert to int before parsing because datetime might be read as float
         return dateutil.parser.parse(str(int(float(s))))
      except:
         raise ValueError("ERROR: Could not convert datetime '%s' to the ISO-8601 standard" % (s))

def run_model( P, radius, lon, lat, verbose = False ):
   # Populate S with the distances between the given point (lon,lat) and the known points P,
   # grouped by species
   S = {}
   for i in P:
      lon = i[1] # i[1] -> field longitude
      lat = i[2] # i[2] -> field latitude
      try:
         spatial_dist = distance.distance( (args.lon, args.lat), (lon, lat) ).km
      except:
         print( 'WARNING: Failed to calculate the distance between ({},{}) and ({},{}), skipping...'.format( args.lon, args.lat, lon, lat ) )
         continue
      if spatial_dist <= args.radius:
         spatial_dist /= args.radius # take off the unit and normalize between [0,1]
         if len( i ) > 3 and args.date is not None:
            day_of_the_year = str_to_datetime( i[3] ).timetuple().tm_yday # i[3] -> field observation date
            day_of_the_year_given = args.date.timetuple().tm_yday

            seasonal_dist = abs( day_of_the_year - day_of_the_year_given )/366.0 # 366.0 is the maximum number of days in a year; normalize between [0,1]
            dist = (1.0 - args.date_weight) * spatial_dist + args.date_weight * seasonal_dist
         else:
            dist = spatial_dist # use spatial distance only
         S.setdefault( i[0], [] ).append( dist ) # i[0] -> field species name

   if len( S ) < args.minimum_instances: # Not enough instance left to build a model
      if verbose:
         print( 'WARNING: Not enough instances to build the model. Minimum is {} but there are only {} within the radius of {}km. This may or may not be an issue, though (it is usually okay if this warning is shown a few times). If you wish you could try adjusting the parameters --minimum-instances and/or --radius.'.format( args.minimum_instances, len( S ), args.radius ) )
      return None

   # Take the k nearest neighbors for each species and calculates the mean distance per species
   Ms = {}
   for i in S:
      Ms[i] = np.mean( np.sort( S[i] )[:args.k] )
   if verbose:
      print( 'Combined distances', {k: v for k, v in sorted(Ms.items(), key=lambda item: item[1])} ) # Sort by distance, ascending

   # Calculate the score for each species (1/dist)
   score_s = {}
   sum = 0.0
   for i in Ms:
      if Ms[i] == 0.0:
         score_s[i] = sys.float_info.max/( len( Ms ) + 1 ) # len(Ms)+1 is to prevent the case where there are multiple points at the same location (dist=0), which would break the sum (cannot be larger than the maximum float!)
      else:
         score_s[i] = 1.0/Ms[i]
      sum += score_s[i]

   if verbose:
      print( '\nScores (1/dist)', {k: v for k, v in sorted(score_s.items(), key=lambda item: item[1], reverse=True)} ) # Sort by 1/dist, descending

   # Normalize the scores (i.e., calculate the probabilities, sum=1)
   for i in score_s:
      score_s[i] = score_s[i]/sum

   # Sorts in descending order (larger probabilities first)
   prob_s_ordered = {k: v for k, v in sorted(score_s.items(), key=lambda item: item[1], reverse=True)} # Sort by probability, descending

   return prob_s_ordered

def main(arguments):

   # Build the main (general) model, which is based on all instances
   probabilities_general = None
   if args.csv is not None and args.weight < 1.0:
      with open( args.csv ) as csvfile:
         P = csv.reader( csvfile )
         probabilities_general = run_model( P, args.radius, args.lon, args.lat, args.verbose )
         if args.verbose:
            print( '\nProbabilities general', probabilities_general )

   # Build the collaborator-specific model, which is based only on the subset of collaborator's instances
   probabilities_collaborator = None
   if args.csv_collaborator is not None and args.weight > 0.0:
      with open( args.csv_collaborator ) as csvfile:
         P = csv.reader( csvfile )
         probabilities_collaborator = run_model( P, args.radius, args.lon, args.lat, args.verbose )
         if args.verbose:
            print( '\nProbabilities collaborator', probabilities_collaborator )

   if probabilities_general is None and probabilities_collaborator is None:
      print( "ERROR: No actual instances to build the model; ensure that at least one non-empty CSV file is given (use -c and/or -cc, and also check if the given radius is large enough)" )
      sys.exit(1)

   if probabilities_collaborator is None:
      probabilities = probabilities_general
   elif probabilities_general is None:
      probabilities = probabilities_collaborator
   else: # Merge is required because there are two sets of probabilities
      # The goal now is to compute a unified list of probabilities by taking a weighted
      # mean of the two lists (actually, dictionaries) of probabilities, i.e., 'probabilities_general' and
      # 'probabilities_collaborator'. The weight is a parameter in [0.0,1.0].
      merged_species = list( {**probabilities_general, **probabilities_collaborator}.keys() )
      merged_probabilities = {}
      for i in merged_species:
         prob = probabilities_general[i] if i in probabilities_general else 0.0
         prob_collaborator = probabilities_collaborator[i] if i in probabilities_collaborator else 0.0
         #print( prob, prob_collaborator )
         weighted_prob = (1.0 - args.weight) * prob + args.weight * prob_collaborator
         if weighted_prob > 0.0:
            merged_probabilities[i] = weighted_prob
      # Overwrite the dictionary 'probabilities' with the sorted-by-probability version of the merged probabilities
      probabilities = {k: v for k, v in sorted(merged_probabilities.items(), key=lambda item: item[1], reverse=True)} # Sort by probability, descending

   print( '\nProbabilities', probabilities )
   print( '\nPredicted species:' )
   max_prob = list( probabilities.values() )[0] # Since the dictionary probabilities is sorted (descending), then the first value is the maximum probability
   for i in probabilities:
      if probabilities[i] == max_prob: # might have multiple species with the same probability
         print( "{} ({:.2f}%)".format( i, probabilities[i]*100 ) )


if __name__ == '__main__':

   parser = argparse.ArgumentParser()

   parser.add_argument("-k", "--k", required=False, type=int, default=2, help="Number of neighbors")
   parser.add_argument("-r", "--radius", required=False, type=float, default=175.0, help="Maximum distance to search for neighbors points (km) [default=175km]. Only the points within that radius are used to build the model.")
   parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Enables verbose mode.")
   parser.add_argument("-lon", "--lon", required=True, type=float, help="Longitude of point of interest, i.e., the longitude of the point whose species prediction is to be computed.")
   parser.add_argument("-lat", "--lat", required=True, type=float, help="Latitude of point of interest, i.e., the latitude of the point whose species prediction is to be computed.")

   parser.add_argument("-c", "--csv", type=str, required=False, help="CSV file of the form 'species,longitude,latitude[,date]' in which the general model will be built upon. If not given, it is expected that the collaborator-specific CSV file will be given in order to (solely) built the model.")
   parser.add_argument("-cc", "--csv-collaborator", type=str, required=False, default=None, help="CSV file of the form 'species,longitude,latitude[,date]' containing only the registers (instances) made by the collaborator for which the prediction is to be computed. The reasoning is that each individual collaborator tends to register only a subset of all species within a specific region featuring a particular environment (sub species distribution); that information can then be exploited to build a model optimized for individual collaborators.")
   parser.add_argument("-w", "--weight", required=False, type=float, default = 0.3, help="How much emphasis to give to collaborator's model probabilities instead of the general model probabilities. Valid values are in [0.0,1.0]. The weight 0.0 means that collaborator probabilities will be in fact discarded whereas 1.0 means that general probabilities will be discarded instead; values in between will produce a weighted mean of the probabilities [default = 0.3].")

   parser.add_argument("-m", "--minimum-instances", required=False, type=int, default=1, help="Minimum number instances to build a model [default=1]. The idea here is to avoid building models that rely on just a few instances, i.e., very low confident models. This is particularly useful to discard collaborator-specific models that would otherwise be built on too few instances and therefore could skew the prediction (when combined with the general model).")
   parser.add_argument("-d", "--date", required=False, type=lambda s: dateutil.parser.parse(s), default=None, help="The date on which the register was made [default=none, meaning that this information will not to used as the temporal distance]. By giving the observation date, the time of the year (year day) is used as the seasonal distance (combined with the spatial distance), taking advantage that the species distribution varies according to the period of the year. The format is YYYY-MM-DD.")
   parser.add_argument("-dw", "--date-weight", required=False, type=float, default = 0.2, help="How much importance to assign to seasonality when calculating the overall distance. Valid values are in [0.0,1.0]. Akin to parameter -w, 0.0 means in practice that the seasonal distance is discarded while 1.0 means that the spatial distance is discarded instead [default=0.2].")

   args = parser.parse_args()

   sys.exit(main(sys.argv[1:]))
