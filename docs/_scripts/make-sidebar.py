import os

pathToLookUpDirectory = "docs/components"
# files = [f for f in os.listdir(pathToLookUpDirectory) if os.path.isfile(f)]
files = [f for f in os.listdir(pathToLookUpDirectory)]

sidebar_file = open('_sidebarPending.md', 'w')
for file in files:
 	if ".md" in file:
 		name = file.split(".md")
 		file = file.replace(" ", "%20")
 		sidebar_file.write( f"* [{name[0]}]({file})\n" )
sidebar_file.close()
