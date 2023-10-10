from alignment_by_row_channels import align

dir = 'video/'
vid1 = 'caroline-choreo.mp4'
vid2 = 'caroline-choreo-2.mp4'

delay = align(vid1, vid2, dir)
print(delay)
