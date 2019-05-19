function individualsBest = computePerfPopulation(individuals)

verbose = 0;

for i= 1 : max(size(individuals))
    
    individual = individuals(i);
    individual.evaluate;
    if verbose
        sprintf('individual %d',i)
        sprintf('score %f',individual.score)
    end
end


[~, ind] = sort([individuals.score],'descend');
individualsBest = individuals(ind);

disp('BEST-SCORE:')
disp(individualsBest(1).score);

disp('WORST-SCORE:')
disp(individualsBest(end).score);

%load('individualBestEver.mat','individualBestEver')
%if individualsBest(1).score > individualBestEver.score
    
    %save('individualBestEver.mat','individualBestEver');
    
%end

disp('AVERAGE-SCORE:')
av_score =0;

for i=1:max(size(individualsBest))
 av_score = av_score + individualsBest(i).score;
end

disp(av_score/max(size(individualsBest)))

    
end