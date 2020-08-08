# All input shaders.
vert = $(wildcard src/shaders/*.vert)
frag = $(wildcard src/shaders/*.frag)

spirv = $(addsuffix .spv,$(vert) $(frag))

default: all

all: $(spirv) cargo out.mp4

cargo:
	cargo run --release

%.vert.spv: %.vert
	glslangValidator -V $< -o $@

%.frag.spv: %.frag
	glslangValidator -V $< -o $@

out.mp4: img/*.png
	ffmpeg -y -r 60 -i img/%03d.png -vcodec libx264 -pix_fmt yuv420p -r 60 $@

clean:
	rm -f $(spirv)

.PHONY: default clean all