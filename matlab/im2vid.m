clc; clear all; close all;

outputVideo = VideoWriter('prop_wifi.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('output','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

prop = 6:13;

for ii = 1:length(prop)    
   img = imread(['output/', imageNames{prop(ii)}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)

%%
outputVideo = VideoWriter('prop_bt.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('output','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

prop = 14:21;

for ii = 1:length(prop)    
   img = imread(['output/', imageNames{prop(ii)}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)

%%
outputVideo = VideoWriter('prop_lora.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('output','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

prop = 22:29;

for ii = 1:length(prop)    
   img = imread(['output/', imageNames{prop(ii)}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)

%%
outputVideo = VideoWriter('ldpl_error_joint.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('output','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

prop = 38:45;

for ii = 1:length(prop)    
   img = imread(['output/', imageNames{prop(ii)}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)

%%
outputVideo = VideoWriter('ldpl_error_ind.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('output','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

prop = 46:53;

for ii = 1:length(prop)    
   img = imread(['output/', imageNames{prop(ii)}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)
%%
outputVideo = VideoWriter('ldpl_error.avi');
outputVideo.FrameRate = 1;
open(outputVideo)
imageNames = dir(fullfile('outputest','*.png'));
imageNames = {imageNames.name}';
imageNames = sort_nat(imageNames);

for ii = 1:length(imageNames)    
   img = imread(['outputest/', imageNames{ii}]);
%    imshow(img);
%    drawnow;
   writeVideo(outputVideo,img)
end

close(outputVideo)