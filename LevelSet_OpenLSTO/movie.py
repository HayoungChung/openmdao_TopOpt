from subprocess import call

cmd = 'ffmpeg -i save/save%03d.png -vcodec mpeg4 save/test.avi'
call(cmd.split())
cmd = 'ffmpeg -i save/test.avi -acodec libmp3lame -ab 192 save/test.mov'
call(cmd.split())
