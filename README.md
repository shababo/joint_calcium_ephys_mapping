# joint_calcium_ephys_mapping

This is a small-scale demo of using Bayesian event inference methods to detect postsynaptic currents.  This also demonstrates joint inference when slower timescale presynaptic information is available (e.g. as would come from calcium imaging). 

To get started, take a look at simulateAndTest which produces a set of passively observed calcium imagining traces and then simulates a post-synaptic cell that is observed via voltage-clamp (note that while the convention in voltage-clamp recordings is for positive deflections to point downwards, the convention in this code is for deflections to-be-detected to point upwards).  In the script, there is a demonstration of inference using just the postsynaptic ephys data, and then there is a demonstration of inference jointly using a subset of the calcium traces and the ephys trace.
