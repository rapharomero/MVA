function [ q ] = annotations( PY,C,s )
%ANNOTATIONS Returns the annotations of the song s
%


set_id=floor(s/15)+1;
song_id=mod(s,15);

As=PY(C==set_id,:);
q=As(:,2*song_id-1:2*song_id);
end

