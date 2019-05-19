function setGlobalIndividualBestEver(IndividualBestEver_)

global IndividualBestEver

IndividualBestEver = Individual;
IndividualBestEver.score = IndividualBestEver_.score;
IndividualBestEver.traj = IndividualBestEver_.traj;
IndividualBestEver.hiddenLayer1 = IndividualBestEver_.hiddenLayer1;
IndividualBestEver.biasHiddenLayer1 = IndividualBestEver_.biasHiddenLayer1;
IndividualBestEver.hiddenLayer2 = IndividualBestEver_.hiddenLayer2;
IndividualBestEver.biasHiddenLayer2 = IndividualBestEver_.biasHiddenLayer2;
IndividualBestEver.r0 = IndividualBestEver_.r0;
IndividualBestEver.v0 = IndividualBestEver_.v0;


return