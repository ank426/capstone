import pandas as pd
import re
from urllib.parse import urlparse, parse_qs

def extract_features_from_stored_xss(stored_xss_df):
    """
    Extract features from stored XSS dataset to match reflective XSS dataset format

    Args:
        stored_xss_df: DataFrame with 'Sentence' and 'XSS' columns

    Returns:
        DataFrame with features compatible with reflective XSS dataset
    """
    # Create a copy of the dataframe to work with
    df = stored_xss_df.copy()

    # Initialize all feature columns with zeros
    reflective_features = [
        'url_length', 'url_special_characters', 'url_tag_script', 'url_tag_iframe',
        'url_attr_src', 'url_event_onload', 'url_event_onmouseover', 'url_cookie',
        'url_number_keywords_param', 'url_number_domain', 'html_tag_script',
        'html_tag_iframe', 'html_tag_meta', 'html_tag_object', 'html_tag_embed',
        'html_tag_link', 'html_tag_svg', 'html_tag_frame', 'html_tag_form',
        'html_tag_div', 'html_tag_style', 'html_tag_img', 'html_tag_input',
        'html_tag_textarea', 'html_attr_action', 'html_attr_background',
        'html_attr_classid', 'html_attr_codebase', 'html_attr_href', 'html_attr_longdesc',
        'html_attr_profile', 'html_attr_src', 'html_attr_usemap', 'html_attr_http-equiv',
        'html_event_onblur', 'html_event_onchange', 'html_event_onclick',
        'html_event_onerror', 'html_event_onfocus', 'html_event_onkeydown',
        'html_event_onkeypress', 'html_event_onkeyup', 'html_event_onload',
        'html_event_onmousedown', 'html_event_onmouseout', 'html_event_onmouseover',
        'html_event_onmouseup', 'html_event_onsubmit', 'html_number_keywords_evil',
        'js_file', 'js_pseudo_protocol', 'js_dom_location', 'js_dom_document',
        'js_prop_cookie', 'js_prop_referrer', 'js_method_write',
        'js_method_getElementsByTagName', 'js_method_getElementById', 'js_method_alert',
        'js_method_eval', 'js_method_fromCharCode', 'js_method_confirm', 'js_min_length',
        'js_min_define_function', 'js_min_function_calls', 'js_string_max_length',
        'html_length'
    ]

    for feature in reflective_features:
        df[feature] = 0

    # Map 'XSS' column to 'Label' format (assuming XSS is already binary 0/1)
    df['Label'] = df['XSS']

    # Process each row
    for idx, row in df.iterrows():
        sentence = str(row['Sentence'])

        # Extract URL-related features
        # Find URLs in the sentence
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, sentence)

        if urls:
            longest_url = max(urls, key=len)
            df.at[idx, 'url_length'] = len(longest_url)

            # Count special characters in URL
            url_special_chars = sum(1 for c in longest_url if not c.isalnum() and c not in ':/.')
            df.at[idx, 'url_special_characters'] = url_special_chars

            # Parse URL and extract parameters
            try:
                parsed_url = urlparse(longest_url)
                query_params = parse_qs(parsed_url.query)
                df.at[idx, 'url_number_keywords_param'] = len(query_params)

                # Count domains and subdomains
                domain_parts = parsed_url.netloc.split('.')
                df.at[idx, 'url_number_domain'] = len(domain_parts)

                # Check for 'cookie' in URL
                if 'cookie' in longest_url.lower():
                    df.at[idx, 'url_cookie'] = 1
            except:
                pass

        # Check for HTML tags
        html_tags = {
            'script': 'html_tag_script',
            'iframe': 'html_tag_iframe',
            'meta': 'html_tag_meta',
            'object': 'html_tag_object',
            'embed': 'html_tag_embed',
            'link': 'html_tag_link',
            'svg': 'html_tag_svg',
            'frame': 'html_tag_frame',
            'form': 'html_tag_form',
            'div': 'html_tag_div',
            'style': 'html_tag_style',
            'img': 'html_tag_img',
            'input': 'html_tag_input',
            'textarea': 'html_tag_textarea'
        }

        for tag, feature in html_tags.items():
            pattern = r'<\s*' + tag + r'[^>]*>'
            if re.search(pattern, sentence, re.IGNORECASE):
                df.at[idx, feature] = 1

                # Also check for script/iframe in URL context
                if tag == 'script' and urls:
                    for url in urls:
                        if re.search(pattern, url, re.IGNORECASE):
                            df.at[idx, 'url_tag_script'] = 1

                if tag == 'iframe' and urls:
                    for url in urls:
                        if re.search(pattern, url, re.IGNORECASE):
                            df.at[idx, 'url_tag_iframe'] = 1

        # Check for HTML attributes
        html_attrs = {
            'action': 'html_attr_action',
            'background': 'html_attr_background',
            'classid': 'html_attr_classid',
            'codebase': 'html_attr_codebase',
            'href': 'html_attr_href',
            'longdesc': 'html_attr_longdesc',
            'profile': 'html_attr_profile',
            'src': 'html_attr_src',
            'usemap': 'html_attr_usemap',
            'http-equiv': 'html_attr_http-equiv'
        }

        for attr, feature in html_attrs.items():
            pattern = r'\s' + attr + r'\s*='
            if re.search(pattern, sentence, re.IGNORECASE):
                df.at[idx, feature] = 1

                # Check for src in URL context
                if attr == 'src' and urls:
                    for url in urls:
                        if re.search(pattern, url, re.IGNORECASE):
                            df.at[idx, 'url_attr_src'] = 1

        # Check for HTML events
        html_events = {
            'onblur': 'html_event_onblur',
            'onchange': 'html_event_onchange',
            'onclick': 'html_event_onclick',
            'onerror': 'html_event_onerror',
            'onfocus': 'html_event_onfocus',
            'onkeydown': 'html_event_onkeydown',
            'onkeypress': 'html_event_onkeypress',
            'onkeyup': 'html_event_onkeyup',
            'onload': 'html_event_onload',
            'onmousedown': 'html_event_onmousedown',
            'onmouseout': 'html_event_onmouseout',
            'onmouseover': 'html_event_onmouseover',
            'onmouseup': 'html_event_onmouseup',
            'onsubmit': 'html_event_onsubmit'
        }

        for event, feature in html_events.items():
            pattern = r'\s' + event + r'\s*='
            if re.search(pattern, sentence, re.IGNORECASE):
                df.at[idx, feature] = 1

                # Check for onload/onmouseover in URL context
                if event == 'onload' and urls:
                    for url in urls:
                        if re.search(pattern, url, re.IGNORECASE):
                            df.at[idx, 'url_event_onload'] = 1

                if event == 'onmouseover' and urls:
                    for url in urls:
                        if re.search(pattern, url, re.IGNORECASE):
                            df.at[idx, 'url_event_onmouseover'] = 1

        # Count evil keywords (common XSS keywords)
        evil_keywords = ['script', 'alert', 'eval', 'javascript', 'onerror', 'onload',
                         'document.cookie', 'fromCharCode', 'iframe', 'svg', 'prompt']
        count = sum(1 for keyword in evil_keywords if keyword.lower() in sentence.lower())
        df.at[idx, 'html_number_keywords_evil'] = count

        # JavaScript features
        js_features = {
            'javascript:': 'js_pseudo_protocol',
            'location': 'js_dom_location',
            'document': 'js_dom_document',
            'cookie': 'js_prop_cookie',
            'referrer': 'js_prop_referrer',
            '.write(': 'js_method_write',
            'getElementsByTagName': 'js_method_getElementsByTagName',
            'getElementById': 'js_method_getElementById',
            'alert(': 'js_method_alert',
            'eval(': 'js_method_eval',
            'fromCharCode': 'js_method_fromCharCode',
            'confirm(': 'js_method_confirm'
        }

        for keyword, feature in js_features.items():
            if keyword.lower() in sentence.lower():
                df.at[idx, feature] = 1

        # External JavaScript file check
        if re.search(r'src\s*=\s*["\'][^"\']*\.js["\']', sentence, re.IGNORECASE):
            df.at[idx, 'js_file'] = 1

        # JavaScript metrics
        # Check for JavaScript code chunks
        js_chunks = re.findall(r'<script[^>]*>(.*?)</script>', sentence, re.DOTALL | re.IGNORECASE)
        js_chunks += re.findall(r'javascript:(.*?)(?=\s|$|\"|\')', sentence, re.IGNORECASE)

        js_length = 0
        func_declarations = 0
        func_calls = 0
        max_string_length = 0

        for chunk in js_chunks:
            js_length += len(chunk)

            # Count function declarations
            declarations = len(re.findall(r'function\s+\w+\s*\(', chunk))
            declarations += len(re.findall(r'(var|let|const)\s+\w+\s*=\s*function', chunk))
            declarations += len(re.findall(r'(var|let|const)\s+\w+\s*=\s*\(.*?\)\s*=>', chunk))
            func_declarations += declarations

            # Count function calls
            calls = len(re.findall(r'\w+\s*\(', chunk)) - declarations
            func_calls += max(0, calls)

            # Find max string length
            strings = re.findall(r'[\'"].*?[\'"]', chunk)
            if strings:
                max_string = max(strings, key=len)
                max_string_length = max(max_string_length, len(max_string) - 2)  # -2 for quotes

        df.at[idx, 'js_min_length'] = js_length
        df.at[idx, 'js_min_define_function'] = func_declarations
        df.at[idx, 'js_min_function_calls'] = func_calls
        df.at[idx, 'js_string_max_length'] = max_string_length

        # HTML length (all content between tags)
        html_content = re.findall(r'<[^>]*>.*?</[^>]*>', sentence, re.DOTALL)
        html_content_length = sum(len(h) for h in html_content)
        df.at[idx, 'html_length'] = html_content_length or len(sentence)

    # Drop the original 'Sentence' and 'XSS' columns if desired
    result_df = df.drop(columns=['Sentence', 'XSS', 'Unnamed: 0'])

    return result_df

def combine_datasets(reflective_df, stored_df):
    """
    Combine the reflective and processed stored XSS datasets

    Args:
        reflective_df: DataFrame with reflective XSS data
        stored_df: DataFrame with stored XSS data (original)

    Returns:
        Combined DataFrame
    """
    # Process the stored XSS dataset
    processed_stored_df = extract_features_from_stored_xss(stored_df)

    # Add a source column to track the origin of each sample
    reflective_df['source'] = 'reflective'
    processed_stored_df['source'] = 'stored'

    # Combine the datasets
    combined_df = pd.concat([reflective_df, processed_stored_df], ignore_index=True)

    return combined_df

# Example usage:
reflective_df = pd.read_csv('data/Data_66_featurs.csv').drop_duplicates()
stored_df = pd.read_csv('data/XSSDataset.csv', encoding="ISO-8859-1").drop_duplicates()
combined_df = combine_datasets(reflective_df, stored_df)
combined_df.to_csv('data/combined_xss_dataset.csv', index=False)
