function [ x ] = firstechoes( source,D,S,M,c )
%FIRSTECHOES First echoes in the reverberation system
%   Parameters
%       D : dimensions of the room
%       S : position of the source
%       M : position of the microphone
%       c : speed of sound
%   Returns
%       x : signal with first order echoes


% Image sources
I = [[-S(1) S(2) S(3)];
    [S(1) -S(2) S(3)];
    [S(1) S(2) -S(3)];
    [2*D(1)-S(1) S(2) S(3)];
    [S(1) 2*D(2)-S(2) S(3)];
    [S(1) S(2) 2*D(3)-S(3)]];

% Distances between mic and sources
distances = [norm(S-M);
            norm(I(1,:)-M);
            norm(I(2,:)-M);
            norm(I(3,:)-M);
            norm(I(4,:)-M);
            norm(I(5,:)-M);
            norm(I(6,:)-M);];
% attenuations
att = min(1,1./distances);
% 
delays = round(distances/c)

% Filter design
filtersize = max(delays);
a = 1;
b = zeros(filtersize,1);
for k=1:length(delays)
    b(delays(k))=att(k);
end

x = filter(b,a,source);

end