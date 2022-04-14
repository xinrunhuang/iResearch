# encoding=utf8
import sys
import io
import json
from imp import reload

if sys.version_info[0]<3:
  reload(sys)
  sys.setdefaultencoding('utf8')
from ScienceParse import ScienceParse

class ScienceParseReader:
  """
    This class reads the output of the science parse library and stores it in theScienceParseclass
  """

  @staticmethod
  def read_science_parse(paperid, title, abstract, scienceparse_dir):
    scienceparse_file = io.open('%s%s.pdf.json'%(scienceparse_dir,paperid), "r", encoding="utf8")
    scienceparse_str = scienceparse_file.read()
    scienceparse_data = json.loads(scienceparse_str)

    #read scienceparse
    scienceparse_map = {}

    sections = {}
    reference_years = {}
    reference_titles = {}
    reference_venues = {}
    reference_mention_contexts = {}
    reference_num_mentions = {}

    name = scienceparse_data["name"]
    metadata = scienceparse_data["metadata"]

    if metadata["sections"] is not None:
      for sectid in range(len(metadata["sections"])):
        heading = metadata["sections"][sectid]["heading"]
        text = metadata["sections"][sectid]["text"]
        sections[str(heading)] = text

    for refid in range(len(metadata["references"])):
      reference_titles[refid] = metadata["references"][refid]["title"]
      reference_years[refid] = metadata["references"][refid]["year"]
      reference_venues[refid] = metadata["references"][refid]["venue"]

    for menid in range(len(metadata["referenceMentions"])):
      refid = metadata["referenceMentions"][menid]["referenceID"]
      context = metadata["referenceMentions"][menid]["context"]
      oldContext = reference_mention_contexts.get(refid, "")
      reference_mention_contexts[refid] = oldContext + "\t" + context
      count = reference_num_mentions.get(refid, 0)
      reference_num_mentions[refid] = count + 1

    authors = metadata["authors"]
    emails = metadata["emails"]
    publication_year = metadata['year'] if metadata['year'] else 2017
    print('pub year: ',publication_year)
    #print(authors)
    #print(emails)

    science_parse = ScienceParse(title, abstract, sections, reference_titles, reference_venues, reference_years,
                                 reference_mention_contexts, reference_num_mentions, authors, emails,pub_year=publication_year)
    return science_parse

  @staticmethod
  def read_science_parse_hxr(scienceparse_dir):
    scienceparse_file = io.open(scienceparse_dir, "r", encoding="utf8")
    scienceparse_str = scienceparse_file.read()
    scienceparse_data = json.loads(scienceparse_str)

    # read scienceparse
    scienceparse_map = {}

    sections = {}
    reference_years = {}
    reference_titles = {}
    reference_venues = {}
    reference_mention_contexts = {}
    reference_num_mentions = {}

    name = scienceparse_data["name"]
    metadata = scienceparse_data["metadata"]
    title = scienceparse_data['metadata']['title']
    abstract = 'Empirically, neural networks that a'
    if metadata["sections"] is not None:
      for sectid in range(len(metadata["sections"])):
        heading = metadata["sections"][sectid]["heading"]
        text = metadata["sections"][sectid]["text"]
        sections[str(heading)] = text

    for refid in range(len(metadata["references"])):
      reference_titles[refid] = metadata["references"][refid]["title"]
      reference_years[refid] = metadata["references"][refid]["year"]
      reference_venues[refid] = metadata["references"][refid]["venue"]

    for menid in range(len(metadata["referenceMentions"])):
      refid = metadata["referenceMentions"][menid]["referenceID"]
      context = metadata["referenceMentions"][menid]["context"]
      oldContext = reference_mention_contexts.get(refid, "")
      reference_mention_contexts[refid] = oldContext + "\t" + context
      count = reference_num_mentions.get(refid, 0)
      reference_num_mentions[refid] = count + 1

    authors = metadata["authors"]
    emails = metadata["emails"]

    # publication_year = metadata['year'] if metadata['year'] else 2017
    # print('pub year: ',publication_year)
    # print(authors)
    # print(emails)

    science_parse = ScienceParse(title, abstract, sections, reference_titles, reference_venues, reference_years,
                                 reference_mention_contexts, reference_num_mentions, authors, emails)
    return science_parse
