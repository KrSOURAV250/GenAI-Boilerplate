"""
Output parsers for GenAI pipelines (str / json / xml).

Usage::

    from common_module.parse_mill.str_refinery import StrParser
    from common_module.parse_mill.json_refinery import JsonParser
    from common_module.parse_mill.xml_refinery import XmlParser

    text = StrParser.get().invoke(llm_response)
    data = JsonParser.get().invoke(llm_response)
"""
