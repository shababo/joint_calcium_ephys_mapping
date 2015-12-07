# joint_calcium_ephys_mapping

This is a small-scale demo of using Bayesian event methods to perform "mapping" of connection strengths from a passively observed calcium and ephys experiment.  This code is a particular extension of some of the code in the related repo: https://github.com/shababo/psc-detection

To get started, take a look at simulateAndTest which produces a set of passively observed calcium imagining traces and then simulates a post-synaptic cell that is observed via current-clamp.  

Inference is then performed jointly on a subset of the calcium traces and the ephys trace.
