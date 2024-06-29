import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.Bitmap(img).write('cbox.exr')